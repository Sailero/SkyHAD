"""Streaming video utility; fake encoders test the public frame/lifecycle contract."""
import hashlib
import copy
import importlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest


class FrameEnv:
    render_mode = 'rgb_array'
    metadata = {'render_fps': 14}
    def __init__(self):
        self.num_cycles = 0
        self.closed = False
        self.render_calls = 0
        self.frame = np.full((32, 48, 3), 17, np.uint8)
    def render(self):
        self.render_calls += 1
        return self.frame
    def close(self): self.closed = True


class DigestWriter:
    """An external encoder stand-in stores only digests, never frame arrays."""
    def __init__(self, path, options):
        self.path, self.options = path, options
        self.digests = []
        self.shapes = []
        self.closed_count = 0
    def append_data(self, frame):
        self.digests.append(hashlib.sha256(frame.tobytes()).hexdigest())
        self.shapes.append(frame.shape)
    def close(self): self.closed_count += 1


@pytest.fixture
def writers(monkeypatch):
    result=[]
    def get_writer(path, **options):
        writer=DigestWriter(path, options)
        result.append(writer)
        return writer
    monkeypatch.setitem(sys.modules, 'imageio.v2', types.SimpleNamespace(get_writer=get_writer))
    monkeypatch.setitem(sys.modules, 'imageio_ffmpeg', types.SimpleNamespace())
    return result


def recorder_type():
    return importlib.import_module('had_env.render.video').VideoRecorder


def test_capture_streams_current_rgb_without_advancing_or_mutating_env(writers, tmp_path):
    env=FrameEnv()
    first_frame=env.frame
    original=first_frame.copy()
    with recorder_type()(env,tmp_path/'nested'/'episode.mp4') as recorder:
        recorder.capture()
        env.frame=np.full((32,48,3),29,np.uint8)
        recorder.capture()
        assert writers[0].digests == [hashlib.sha256(original.tobytes()).hexdigest(),
                                      hashlib.sha256(env.frame.tobytes()).hexdigest()]
        assert len(writers) == 1 and writers[0].closed_count == 0
    assert env.num_cycles == 0 and not env.closed and env.render_calls == 2
    np.testing.assert_array_equal(first_frame,original)
    assert np.all(env.frame == 29)
    assert writers[0].closed_count == 1
    assert writers[0].options['fps'] == 14


def test_capture_requires_active_context_and_close_is_idempotent(writers,tmp_path):
    recorder=recorder_type()(FrameEnv(),tmp_path/'episode.mp4',fps=7)
    with pytest.raises(RuntimeError,match='enter|context|open'): recorder.capture()
    with recorder:
        with pytest.raises(RuntimeError,match='already|open'): recorder.__enter__()
        recorder.capture()
    recorder.close()
    with pytest.raises(RuntimeError,match='closed'): recorder.capture()
    with pytest.raises(RuntimeError,match='closed'): recorder.__enter__()
    assert writers[0].closed_count == 1 and writers[0].options['fps'] == 7


@pytest.mark.parametrize('mode',[None,'human'])
def test_recording_rejects_non_rgb_render_mode_before_opening_writer(writers,tmp_path,mode):
    env=FrameEnv()
    env.render_mode=mode
    with pytest.raises(ValueError,match='rgb_array'): recorder_type()(env,tmp_path/'episode.mp4')
    assert writers == [] and env.render_calls == 0


@pytest.mark.parametrize('fps',[0,-1,math.nan,math.inf,True,'30'])
def test_invalid_fps_does_not_open_encoder(writers,tmp_path,fps):
    with pytest.raises(ValueError,match='fps'): recorder_type()(FrameEnv(),tmp_path/'episode.mp4',fps=fps)
    assert writers == []


def test_body_failure_closes_writer_without_closing_owned_environment(writers,tmp_path):
    env=FrameEnv()
    with pytest.raises(LookupError,match='caller failure'):
        with recorder_type()(env,tmp_path/'episode.mp4') as recorder:
            recorder.capture()
            raise LookupError('caller failure')
    assert writers[0].closed_count == 1 and not env.closed


def test_encoder_failure_closes_stream(writers,tmp_path):
    with pytest.raises(OSError,match='encoder failure'):
        with recorder_type()(FrameEnv(),tmp_path/'episode.mp4') as recorder:
            def fail(frame): raise OSError('encoder failure')
            writers[0].append_data=fail
            recorder.capture()
    assert writers[0].closed_count == 1


@pytest.mark.parametrize('frame',[None,np.zeros((5,5),np.uint8),np.zeros((5,5,4),np.uint8),np.zeros((5,5,3),np.float32)])
def test_capture_rejects_non_rgb_uint8_frames(writers,tmp_path,frame):
    env=FrameEnv()
    env.frame=frame
    with recorder_type()(env,tmp_path/'episode.mp4') as recorder:
        with pytest.raises(ValueError,match='RGB|rgb|uint8'): recorder.capture()
    assert writers[0].digests == []


def test_optional_record_dependencies_load_only_when_context_opens(monkeypatch,tmp_path):
    module=importlib.import_module('had_env.render.video')
    imported=[]
    def missing(name):
        imported.append(name)
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(module,'import_module',missing)
    recorder=module.VideoRecorder(FrameEnv(),tmp_path/'episode.mp4')
    assert imported == []
    with pytest.raises(RuntimeError,match=r'\[record\]'): recorder.__enter__()
    recorder.close()


def example_module():
    path=Path(__file__).resolve().parents[1]/'examples'/'rendering.py'
    spec=importlib.util.spec_from_file_location('rendering_example',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('video',[False,True])
def test_rendering_example_owns_action_loop_and_closes_environment(writers,tmp_path,monkeypatch,video):
    example=example_module()
    created=[]
    class LoopEnv(FrameEnv):
        def __init__(self,config):
            super().__init__()
            self.config=config
            self.render_mode=config['render_mode']
            self.max_cycles=config['max_cycles']
            self.agents=['red_0','blue_0']
            self.mode=None
            self.seed=None
        def set_view_mode(self,mode): self.mode=mode
        def reset(self,seed=None): self.seed=seed
        def action_space(self,agent): return types.SimpleNamespace(sample=lambda:0)
        def step(self,actions):
            assert set(actions) == set(self.agents)
            self.num_cycles+=1
            self.frame=np.full((32,48,3),self.num_cycles,np.uint8)
            if self.num_cycles >= self.max_cycles: self.agents=[]
    def factory(*args,**config):
        env=LoopEnv(config)
        created.append(env)
        return env
    monkeypatch.setattr(example,'make_env',factory)
    args=['--view','3d','--max-steps','2','--seed','7']
    if video: args+=['--video',str(tmp_path/'example.mp4')]
    example.main(args)
    env=created[0]
    assert env.closed and env.num_cycles == 2 and env.mode == '3d' and env.seed == 7
    assert env.config['env_agent_type'] == 'UAV_fixedwing'
    assert env.render_mode == ('rgb_array' if video else 'human')
    if video: assert len(writers[0].digests) == 3 and writers[0].closed_count == 1
    else: assert writers == []


def test_real_environment_capture_preserves_physics_and_rng(writers,tmp_path,monkeypatch):
    monkeypatch.setenv('SDL_VIDEODRIVER','dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER','dummy')
    pytest.importorskip('pygame')
    from had_env import make_env
    env=make_env('defense',red_count=1,blue_count=1,env_agent_type='UAV_fixedwing',
                 spatial_dim=3,render_mode='rgb_array',max_cycles=2)
    try:
        env.reset(seed=11)
        state=env.state().copy()
        rng=copy.deepcopy(env.np_random.bit_generator.state)
        agents=list(env.agents)
        with recorder_type()(env,tmp_path/'native.mp4') as recorder:
            recorder.capture()
            recorder.capture()
        np.testing.assert_array_equal(env.state(),state)
        assert env.np_random.bit_generator.state == rng
        assert env.agents == agents and env.num_cycles == 0
        assert len(writers[0].digests) == 2 and writers[0].digests[0] == writers[0].digests[1]
    finally: env.close()


def test_odd_rgb_dimensions_pad_edge_pixels_without_mutating_source(writers,tmp_path):
    env=FrameEnv()
    env.frame=np.arange(15*17*3,dtype=np.uint8).reshape(15,17,3)
    original=env.frame.copy()
    expected=np.empty((16,18,3),dtype=np.uint8)
    expected[:15,:17]=original
    expected[15,:17]=original[14]
    expected[:15,17]=original[:,16]
    expected[15,17]=original[14,16]
    with recorder_type()(env,tmp_path/'odd.mp4') as recorder:
        recorder.capture()
    assert writers[0].shapes == [(16,18,3)]
    assert writers[0].digests == [hashlib.sha256(expected.tobytes()).hexdigest()]
    np.testing.assert_array_equal(env.frame,original)


def test_real_ffmpeg_odd_rgb_video_decodes_two_padded_frames(tmp_path):
    imageio=pytest.importorskip('imageio.v2')
    ffmpeg=pytest.importorskip('imageio_ffmpeg')
    try:
        ffmpeg.get_ffmpeg_exe()
    except RuntimeError as exc:
        pytest.skip(f'FFmpeg executable unavailable: {exc}')
    env=FrameEnv()
    env.frame=np.full((15,17,3),[20,80,140],dtype=np.uint8)
    original=env.frame.copy()
    output=tmp_path/'odd-real.mp4'
    with recorder_type()(env,output,fps=6) as recorder:
        recorder.capture()
        recorder.capture()
    assert output.stat().st_size > 0
    with imageio.get_reader(output) as reader:
        assert reader.count_frames() == 2
        assert reader.get_data(0).shape == (16,18,3)
        assert reader.get_data(1).shape == (16,18,3)
    np.testing.assert_array_equal(env.frame,original)
