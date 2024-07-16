import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from torch import nn

sample_rate = 16000
n_fft = 640
hop_length = 160


class ConvSTFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()

        kernel_real, kernel_imag = self.get_kernel(n_fft)

        self.conv_real = torch.nn.Conv1d(
            1, n_fft // 2 + 1, n_fft, stride=hop_length, padding=n_fft // 2, bias=False
        )
        self.conv_real.weight.data = kernel_real.unsqueeze(1)
        self.conv_real.requires_grad_(False)

        self.conv_imag = torch.nn.Conv1d(
            1, n_fft // 2 + 1, n_fft, stride=hop_length, padding=n_fft // 2, bias=False
        )
        self.conv_imag.weight.data = kernel_imag.unsqueeze(1)
        self.conv_imag.requires_grad_(False)

    @staticmethod
    def get_kernel(n_fft):
        window = torch.hann_window(n_fft)

        fft_matrix = torch.fft.rfft(torch.eye(n_fft)).T
        fft_matrix = fft_matrix * window.unsqueeze(0)
        kernel_real, kernel_imag = fft_matrix.real, fft_matrix.imag
        # print(fft_matrix.shape)
        # plt.plot(kernel_real[2])
        # plt.plot(kernel_imag[2])
        # plt.show()
        return kernel_real, kernel_imag

    def forward(self, x):
        real = self.conv_real(x)
        imag = self.conv_imag(x)
        complex = torch.complex(real, imag)

        return complex


if __name__ == "__main__":

    def load_test_wav():
        wav, sr = torchaudio.load("test.wav")
        wav = wav.mean(0, keepdim=True)
        if sr != sample_rate:
            wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
        return wav

    def test():
        wav = load_test_wav().unsqueeze(0)
        conv_stft = ConvSTFT(n_fft, hop_length)
        stft_complex = conv_stft(wav)
        # stft_real, stft_imag = stft_complex.real, stft_complex.imag
        stft_mag = torch.abs(stft_complex)
        print(wav.shape, stft_mag.shape)
        plt.imshow(
            torch.log(stft_mag[0] + 1e-6).detach().cpu(), origin="lower", aspect="auto"
        )
        plt.show()

    def test_onnx():
        conv_stft = ConvSTFT(n_fft, hop_length)
        input = torch.randn(4, 1, 16000)

        export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
        onnx_program = torch.onnx.dynamo_export(
            conv_stft, input, export_options=export_options
        )
        onnx_program.save("model.onnx")

        ort_session = ort.InferenceSession("model.onnx")
        input_name = ort_session.get_inputs()[0].name
        wav = load_test_wav().unsqueeze(0).numpy()
        ort_output = ort_session.run(None, {input_name: wav})
        print(ort_output[0])
        ort_output_mag = np.sqrt(
            ort_output[0][..., 0] ** 2 + ort_output[0][..., 1] ** 2
        )
        plt.imshow(np.log(ort_output_mag[0] + 1e-6), origin="lower", aspect="auto")
        plt.show()

    test_onnx()
