import os
print("Current working directory: {0}".format(os.getcwd()))

import uuid
from utils.fixseed import fixseed
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
from sample.generate import save_multiple_samples, load_dataset, construct_template_variables


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class MotionGenerator:
    def __init__(self, model_path='./save/humanml_trans_enc_512/model000200000.pt'):
        args = Namespace(model_path=model_path, cuda=True, device=0, seed=10,
                         batch_size=64, output_dir='', num_samples=10,
                         num_repetitions=1, guidance_param=2.5,
                         motion_length=6.0, input_text='', action_file='',
                         text_prompt='This will be replaced while generating',
                         action_name='', dataset='humanml', data_dir='',
                         arch='trans_enc', emb_trans_dec=False, layers=8,
                         latent_dim=512, cond_mask_prob=0.1, lambda_rcxyz=0.0,
                         lambda_vel=0.0, lambda_fc=0.0, unconstrained=False,
                         noise_schedule='cosine', diffusion_steps=50,
                         sigma_small=True)
        fixseed(args.seed)
        out_path = args.output_dir
        name = os.path.basename(os.path.dirname(args.model_path))
        niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
        max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
        fps = 12.5 if args.dataset == 'kit' else 20
        n_frames = min(max_frames, int(args.motion_length*fps))
        is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
        print('is using data?', is_using_data)
        dist_util.setup_dist(args.device)
        args.num_samples = 1
        assert args.num_samples <= args.batch_size, \
            f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
        # So why do we need this check? In order to protect GPU from a memory overload in the following line.
        # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
        # If it doesn't, and you still want to sample more prompts, run this script with different seeds
        # (specify through the --seed flag)
        args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

        print('Loading dataset...')
        data = load_dataset(args, max_frames, n_frames)
        total_num_samples = args.num_samples * args.num_repetitions

        print("Creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(args, data)

        print(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict)

        if args.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
        model.to(dist_util.dev())
        model.eval()  # disable random masking

        self.args = args
        self.diffusion = diffusion
        self.model = model
        self.max_frames = max_frames
        self.n_frames = n_frames
        self.data = data
        self.fps = fps

    def generate(self, text_prompt):
        args = self.args
        diffusion = self.diffusion
        model = self.model
        max_frames = self.max_frames
        n_frames = self.n_frames
        data = self.data
        fps = self.fps

        total_num_samples = args.num_samples * args.num_repetitions

        texts = [text_prompt]
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs = collate(collate_args)

        all_motions = []
        all_lengths = []
        all_text = []

        for rep_i in range(args.num_repetitions):
            print(f'### Sampling [repetitions #{rep_i}]')

            # add CFG scale to batch
            if args.guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

            sample_fn = diffusion.p_sample_loop

            sample = sample_fn(
                model,
                # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
                (args.batch_size, model.njoints, model.nfeats, max_frames),  # BUG FIX
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
            rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
            sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                get_rotations_back=False)

            if args.unconstrained:
                all_text += ['unconstrained'] * args.num_samples
            else:
                text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
                all_text += model_kwargs['y'][text_key]

            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

            print(f"created {len(all_motions) * args.batch_size} samples")


        all_motions = np.concatenate(all_motions, axis=0)
        all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
        all_text = all_text[:total_num_samples]
        all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

        # TODO fix this and define an output file
        out_path_prefix = './output'
        while True:
            out_path = os.path.join(out_path_prefix, str(uuid.uuid1()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
                break
        
        print(f"saving visualizations to [{out_path}]...")
        skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

        sample_files = []
        num_samples_in_out_file = 7

        sample_print_template, row_print_template, all_print_template, \
        sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

        for sample_i in range(args.num_samples):
            rep_files = []
            for rep_i in range(args.num_repetitions):
                caption = all_text[rep_i*args.batch_size + sample_i]
                length = all_lengths[rep_i*args.batch_size + sample_i]
                motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
                save_file = sample_file_template.format(sample_i, rep_i)
                print(sample_print_template.format(caption, sample_i, rep_i, save_file))
                animation_save_path = os.path.join(out_path, save_file)
                plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
                # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
                rep_files.append(animation_save_path)

            sample_files = save_multiple_samples(args, out_path,
                                                  row_print_template, all_print_template, row_file_template, all_file_template,
                                                  caption, num_samples_in_out_file, rep_files, sample_files, sample_i)

        abs_path = os.path.join(os.path.abspath(out_path), 'sample00.mp4')
        print(f'[Done] Results are at [{abs_path}]')
        return abs_path


def main():
    motion_generator = MotionGenerator(
        model_path='./save/humanml_trans_enc_512/model000200000.pt'
        )
    video_mp4_file_absolute_path = motion_generator.generate('man boxing')
