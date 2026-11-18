"""Preview exactly the task-selected terrain types using the training generator."""
import argparse
import copy
import numpy as np
import isaacgym
from isaacgym import gymapi
import torch
from robot_sim.registry import make_config, make_env, task_classes
from robot_sim.simulation.terrain import Terrain, TERRAIN_TYPES, resolve_terrain


def preview(args):
    cfg=make_config(args.task)
    if args.terrain_ids is not None:
        cfg.terrain.selected_ids=args.terrain_ids
        cfg.terrain.weights=None
    resolve_terrain(cfg.terrain)
    ids=cfg.terrain.resolved_ids
    source_rows=cfg.terrain.num_rows
    levels=args.levels or sorted(set(np.rint(np.linspace(0,source_rows-1,4)).astype(int).tolist()))
    if any(i<0 or i>=source_rows for i in levels):
        raise ValueError('Preview levels must lie within task curriculum range')
    cfg.terrain.preview_difficulties=[i/source_rows for i in levels]
    cfg.terrain.num_rows=len(levels)
    cfg.terrain.num_cols=len(ids)
    cfg.terrain.max_init_terrain_level=len(levels)-1
    cfg.terrain.curriculum=True
    cfg.terrain.weights=None
    cfg.env.num_envs=len(levels)*len(ids)
    cfg.commands.curriculum=False
    task_class,_=task_classes(args.task)

    class PreviewTerrain(Terrain):
        def curiculum(self):
            self.column_ids=np.array(self.cfg.resolved_ids)
            for col, terrain_id in enumerate(self.column_ids):
                lower=0 if terrain_id==0 else self.proportions[terrain_id-1]
                width=self.proportions[terrain_id]-lower
                for row,difficulty in enumerate(self.cfg.preview_difficulties):
                    fraction=0.25 if row%2==0 else 0.75
                    self.add_terrain_to_map(self.make_terrain(lower+fraction*width,difficulty),row,col)

    class PreviewEnv(task_class):
        def create_sim(self):
            self.up_axis_idx=2
            self.sim=self.gym.create_sim(self.sim_device_id,self.graphics_device_id,self.physics_engine,self.sim_params)
            if self.sim is None:
                raise RuntimeError('Isaac Gym simulation creation failed')
            self.terrain=PreviewTerrain(self.cfg.terrain,self.num_envs)
            self._create_trimesh()
            self._create_envs()

        def _get_env_origins(self):
            super()._get_env_origins()
            self.terrain_types=torch.arange(self.num_envs,device=self.device)//self.cfg.terrain.num_rows
            self.terrain_levels=torch.arange(self.num_envs,device=self.device)%self.cfg.terrain.num_rows
            self.env_origins[:]=self.terrain_origins[self.terrain_levels,self.terrain_types]

    # Local class only; never replace task registration.
    from robot_sim.registry import register_task
    register_task('_preview',PreviewEnv,type(cfg))
    env=make_env('_preview',cfg,args.device,args.headless)
    try:
        env.reset()
        for terrain_id in ids:
            print(terrain_id,TERRAIN_TYPES[terrain_id],'levels',levels,'difficulty',cfg.terrain.preview_difficulties)
        if env.viewer:
            length=cfg.terrain.num_rows*cfg.terrain.terrain_length
            width=cfg.terrain.num_cols*cfg.terrain.terrain_width
            env.set_camera([length/2,-12,max(length,width)*0.8],[length/2,width/2,0])
        frames=0
        while not args.frames or frames<args.frames:
            if env.viewer:
                if env.gym.query_viewer_has_closed(env.viewer): break
                env.gym.poll_viewer_events(env.viewer)
                env.gym.step_graphics(env.sim)
                env.gym.draw_viewer(env.viewer,env.sim,True)
                env.gym.sync_frame_time(env.sim)
            frames+=1
        if args.screenshot:
            if not env.viewer: raise ValueError('Screenshot requires a viewer')
            env.gym.write_viewer_image_to_file(env.viewer,args.screenshot)
        print('Preview complete:',frames,'frames,',env.num_envs,'terrain tiles')
    finally: env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',default='wheelleg')
    parser.add_argument('--terrain-ids',nargs='+',type=int)
    parser.add_argument('--levels',nargs='+',type=int)
    parser.add_argument('--frames',type=int,default=0)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--screenshot')
    args=parser.parse_args()
    if args.headless and args.frames<=0: parser.error('--headless requires --frames > 0')
    preview(args)
