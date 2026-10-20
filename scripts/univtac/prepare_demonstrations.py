"""Build action-aligned visual, tactile and motion context from UniVTAC episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from sim.envs.univtac.autonomous_operation import matrix_rotvec, quaternion_matrix
from sim.envs.univtac.tactile_history import VIEWS, motion_features, select_times
from sim.envs.univtac.trace import write_json

def segment_bounds(atom_ids, tags):
    starts=[i for i in range(len(tags)) if i==0 or atom_ids[i]!=atom_ids[i-1] or tags[i]!=tags[i-1]]
    return [(max(0,a-1),b-1) for a,b in zip(starts,starts[1:]+[len(tags)])]


def strip(frames, selected, steps, path, label):
    w,h=frames[0].size
    canvas=Image.new('RGB',(w*len(selected),h+30),'white');draw=ImageDraw.Draw(canvas)
    for i,index in enumerate(selected):
        canvas.paste(frames[index],(i*w,30))
        draw.text((i*w+5,8),f'{label} | row {index} | step {int(steps[index])}',fill='black')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)



def decode_images(data):
    """Restore the numeric channel order used by UniVTAC's JPEG producer."""
    images = []
    for value in data.ravel():
        encoded = np.frombuffer(value, dtype=np.uint8) if isinstance(value, (bytes, bytearray)) else value
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Could not decode a demonstration image')
        images.append(image)
    return np.stack(images)


def prepare(raw: Path, out: Path, *, config_path: Path, episodes=(0,1)):
    import h5py
    out.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((raw/'metadata.json').read_text())
    config=yaml.safe_load(config_path.read_text())
    cfg=config['tactile_history']
    task=config['task']
    base={'task_goal':config.get('task_instruction','Insert the held object into the hole.'), 'examples':[],
          'interpretation':'Historical measured expert motion, not recorded move_to commands. Do not copy historical absolute coordinates into the current scene.',
          'robot_convention':'EE is measured panda_hand pose relative to robot base: metres then quaternion wxyz; not OpenETA world gripper-center TCP. Joint vector: seven arm radians then two finger positions in metres.',
          'time_convention':'Frame step values are recorded. Seconds below are nominal step differences / 120 from published collect timing, not an embedded per-frame clock.',
          'outcome_convention':'Success is the official per-episode metadata result; no checker rerun.',
          'unrecorded':'Original actuator/tool commands and force/grasp-state labels are unavailable.'}
    vision_images=[];touch_images=[];touch_content=[]
    for episode in episodes:
        info=metadata[str(episode)]
        if info['result']!='success':
            raise ValueError(f'Episode {episode} is not a successful expert demonstration')
        hdf_root = raw/'hdf5' if (raw/'hdf5').is_dir() else raw
        with h5py.File(hdf_root/f'{episode}.hdf5') as f:
            ee=f['embodiment/ee'][()];joint=f['embodiment/joint'][()];steps=f['step'][()]
            ids=f['atom/id'][()];tags=f['atom/tag'][()]
            keys={n:f'observation/{n}/rgb' for n in VIEWS[:2]}
            keys.update({n:f'tactile/{n}/rgb_marker' for n in VIEWS[2:]})
            pixels={n:decode_images(f[key][()]) for n,key in keys.items()}
            # Native producer encoded its RGB numeric array with cv2.imencode;
            # native imdecode restores those original numbers. Do not swap again.
        if len(steps)<2 or any(len(x)!=len(steps) for x in [ee,joint,ids,tags,*pixels.values()]) or not np.all(np.diff(steps)>0):
            raise ValueError('Image/state lengths do not align')
        frames={n:[Image.fromarray(x) for x in arrays] for n,arrays in pixels.items()}
        recorded=[]
        for i,t in enumerate(steps):
            row={'sample_id':i,'step':int(t),'simulation_time_seconds':float((t-steps[0])/120),'features':{}}
            if i:
                row['features']={n:motion_features(pixels[n][i-1],pixels[n][i],cfg) for n in VIEWS[2:]}
            recorded.append(row)
        bounds=segment_bounds(ids,tags)
        example={'example_id':f'official_episode_{episode}', 'outcome':'success', 'segments':[]}
        for k,(a,b) in enumerate(bounds,1):
            label=f'example {episode} segment {k}'
            def state(i,ee=ee,joint=joint,steps=steps):return {'ee_xyz_wxyz':ee[i].tolist(),'joint':joint[i].tolist(),'recorded_step':int(steps[i])}
            ra=quaternion_matrix(ee[a,3:][[1,2,3,0]]);rb=quaternion_matrix(ee[b,3:][[1,2,3,0]])
            segment={'segment':k,'before':state(a),'after':state(b),
                'measured_motion':{'translation_in_base_mm':((ee[b,:3]-ee[a,:3])*1000).tolist(),
                    'rotation_vector_in_base_rad':matrix_rotvec(rb@ra.T).tolist(),
                    'finger_position_change_m':(joint[b,-2:]-joint[a,-2:]).tolist(),
                    'nominal_duration_s':float((steps[b]-steps[a])/120)},
                'feedback':'Observed state transition; original execution feedback is not separately recorded.',
                'vision_labels':[]}
            for n in VIEWS[:2]:
                name=f'{label} {n} before/after';path=Path('images')/f'e{episode}_s{k}_{n}.png'
                strip(frames[n],[a,b],steps,out/path,n)
                vision_images.append({'label':name,'path':str(path)})
                segment['vision_labels'].append(name)
            selected=[r['sample_id'] for r in select_times(recorded[a:b+1])]
            touch={'example_id':example['example_id'],'segment':k,'recorded_steps':[int(steps[i]) for i in selected],
                   'selection':'segment-end image/motion change, not slip labels','image_labels':[]}
            for n in VIEWS[2:]:
                name=f'{label} {n} history';path=Path('images')/f'e{episode}_s{k}_{n}.png'
                strip(frames[n],selected,steps,out/path,n)
                touch_images.append({'label':name,'path':str(path)});touch['image_labels'].append(name)
            touch_content.append(touch);example['segments'].append(segment)
        base['examples'].append(example)
    base['historical_touch']=touch_content
    write_json(out/'expert_demonstrations.json', {'text':base, 'images':vision_images+touch_images})
    write_json(out/'provenance.json', {
        'dataset':'https://huggingface.co/datasets/byml/UniVTAC',
        'dataset_path':f'isaac51/{task}', 'episode_ids':list(episodes),
        'segmentation':'action boundaries', 'tactile_selection':cfg})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--episodes',type=int,nargs='+',default=[0,1])
    args=parser.parse_args();prepare(args.raw.resolve(),args.output.resolve(),config_path=args.config,episodes=args.episodes)
