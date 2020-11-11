#!/usr/bin/env python3

"""
Usage example::

  python3 pytorch_to_returnn.py \
  --pwg_config mb_melgan.v2.yaml \
  --pwg_checkpoint mb_melgan_models/checkpoint-1000000steps.pkl \
  --features data/features.npy

"""


import argparse
import numpy
import yaml
import wave
import better_exchook
import typing


def main():
    parser = argparse.ArgumentParser(description="MB-MelGAN vocoder")
    parser.add_argument("--features", required=True, help="npy file. via decoder.py --dump_features")
    parser.add_argument("--pwg_config", type=str, help="ParallelWaveGAN config (.yaml)")
    parser.add_argument("--pwg_checkpoint", type=str, help="ParallelWaveGAN checkpoint (.pkl)")
    args = parser.parse_args()

    # from pytorch_to_returnn import trace_torch
    # trace_torch.enable()

    import pytorch_to_returnn.wrapped_import
    pytorch_to_returnn.wrapped_import.LogVerbosity = 4
    from pytorch_to_returnn.verify import verify_torch
    from pytorch_to_returnn.wrapped_import import wrapped_import

    def model_func(wrapped_import):

        if typing.TYPE_CHECKING or not wrapped_import:
            import torch
            from parallel_wavegan import models as pwg_models
            from parallel_wavegan import layers as pwg_layers

        else:
            torch = wrapped_import("torch")
            wrapped_import("parallel_wavegan")
            pwg_models = wrapped_import("parallel_wavegan.models")
            pwg_layers = wrapped_import("parallel_wavegan.layers")

        # Initialize PWG
        pwg_config = yaml.load(open(args.pwg_config), Loader=yaml.Loader)
        pyt_device = torch.device("cpu")
        generator = pwg_models.MelGANGenerator(**pwg_config['generator_params'])
        generator.load_state_dict(
            torch.load(args.pwg_checkpoint, map_location="cpu")["model"]["generator"])
        generator.remove_weight_norm()
        pwg_model = generator.eval().to(pyt_device)
        pwg_pad_fn = torch.nn.ReplicationPad1d(
            pwg_config["generator_params"].get("aux_context_window", 0))
        pwg_pqmf = pwg_layers.PQMF(pwg_config["generator_params"]["out_channels"]).to(pyt_device)

        feature_data = numpy.load(args.features)
        print("Feature shape:", feature_data.shape)

        with torch.no_grad():
            input_features = pwg_pad_fn(torch.from_numpy(feature_data).unsqueeze(0)).to(pyt_device)
            audio_waveform = pwg_pqmf.synthesis(pwg_model(input_features)).view(-1)

        return audio_waveform

    verify_torch(model_func)

    import pytorch_to_returnn._wrapped_mods.torch as torch_
    x = torch_.zeros((1,))
    x = x + 1
    print(x)

    audio_waveform = model_func(wrapped_import)
    audio_waveform = audio_waveform.cpu().numpy()
    audio_raw = numpy.asarray(audio_waveform*(2**15-1), dtype="int16").tobytes()

    out_fn = "out.wav"
    wave_writer = wave.open(out_fn, "wb")
    wave_writer.setnchannels(1)
    wave_writer.setframerate(16000)
    wave_writer.setsampwidth(2)
    wave_writer.writeframes(audio_raw)
    wave_writer.close()
    print("Wrote %s." % out_fn)


if __name__ == "__main__":
    better_exchook.install()
    main()
