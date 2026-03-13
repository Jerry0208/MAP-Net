"""Prepare custom clip folders for MAP-Net test pipeline on Kaggle.

This script converts an input tree such as::

    infer_peng/
      clip_xxx/
        00000.jpeg
        00001.jpeg

into the HazeWorld-like test structure used by configs/dehazers/_base_/datasets/hazeworld.py::

    data/HazeWorld/test/
      hazy/Custom/<clip>_<haze_light>_<beta>/*.jpeg
      gt/Custom/<clip>_<haze_light>_<beta>/*.jpeg
      meta_info_GT_test.txt
      meta_info_tree_GT_test.json

By default GT files are linked to hazy files because inference may not have GT.
"""

import argparse
import json
import os
import os.path as osp
import shutil
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare custom MAP-Net test data layout')
    parser.add_argument('--input-dir', required=True, help='Root dir with clip subfolders')
    parser.add_argument('--output-root', default='data/HazeWorld/test',
                        help='Output split root (contains hazy/ gt/ and meta files)')
    parser.add_argument('--dataset-name', default='Custom',
                        help='Pseudo dataset name under hazy/gt, e.g. Custom')
    parser.add_argument('--haze-light', type=int, default=128,
                        help='Pseudo haze light in folder suffix (integer, e.g. 128)')
    parser.add_argument('--haze-beta', type=float, default=0.005,
                        help='Pseudo haze beta in folder suffix (e.g. 0.005)')
    parser.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg', '.png'],
                        help='Accepted image extensions')
    parser.add_argument('--copy-mode', choices=['copy', 'symlink'], default='symlink',
                        help='How to place files into hazy/gt folders')
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_images(folder, exts):
    files = []
    for ext in exts:
        files.extend(glob(osp.join(folder, f'*{ext}')))
        files.extend(glob(osp.join(folder, f'*{ext.upper()}')))
    files = sorted(set(files))
    return files


def place_file(src, dst, mode):
    if osp.lexists(dst):
        os.remove(dst)
    if mode == 'copy':
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def main():
    args = parse_args()
    input_dir = osp.realpath(args.input_dir)
    output_root = osp.realpath(args.output_root)

    hazy_root = osp.join(output_root, 'hazy', args.dataset_name)
    gt_root = osp.join(output_root, 'gt', args.dataset_name)
    ensure_dir(hazy_root)
    ensure_dir(gt_root)

    clips = sorted([d for d in os.listdir(input_dir)
                    if osp.isdir(osp.join(input_dir, d))])
    if not clips:
        raise RuntimeError(f'No clip folders found in {input_dir}')

    meta_lines = []
    meta_tree = {}

    for clip in clips:
        src_clip_dir = osp.join(input_dir, clip)
        img_files = list_images(src_clip_dir, args.exts)
        if not img_files:
            continue

        dst_folder = f'{clip}_{args.haze_light}_{args.haze_beta}'
        hazy_clip_dir = osp.join(hazy_root, dst_folder)
        gt_clip_dir = osp.join(gt_root, dst_folder)
        ensure_dir(hazy_clip_dir)
        ensure_dir(gt_clip_dir)

        frame_names = []
        for src_path in img_files:
            name = osp.basename(src_path)
            hazy_dst = osp.join(hazy_clip_dir, name)
            gt_dst = osp.join(gt_clip_dir, name)
            place_file(src_path, hazy_dst, args.copy_mode)
            # no GT for inference; mirror as pseudo GT
            if osp.lexists(gt_dst):
                os.remove(gt_dst)
            os.symlink(hazy_dst, gt_dst)
            frame_names.append(name)

        key = f'{args.dataset_name}/{dst_folder}'
        meta_lines.append(f'{key} {len(frame_names)}\n')
        meta_tree[key] = frame_names

    meta_lines.sort()
    with open(osp.join(output_root, 'meta_info_GT_test.txt'), 'w') as f:
        f.writelines(meta_lines)
    with open(osp.join(output_root, 'meta_info_tree_GT_test.json'), 'w') as f:
        json.dump(meta_tree, f, indent=2)

    print(f'Prepared {len(meta_lines)} clips under {output_root}')
    print(f'Meta file: {osp.join(output_root, "meta_info_GT_test.txt")}')
    print(f'Meta tree: {osp.join(output_root, "meta_info_tree_GT_test.json")}')


if __name__ == '__main__':
    main()
