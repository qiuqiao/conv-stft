import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from torch import nn

sample_rate = 16000
hop_length = 160
n_fft = hop_length * 3


def get_matrix(n_fft):
    window = 0.85 * torch.hann_window(n_fft) + 0.15 * torch.hamming_window(n_fft)
    # 单纯的hann窗，istft重建效果不好；单纯的hamming窗，频谱会出现泄露
    # 修改istft的方法也许能解决这个问题，但担心修改后就无法使用onnx导出了
    # 所以用hann与hamming的加权平均

    fft_matrix = torch.fft.rfft(torch.eye(n_fft)).T
    fft_matrix = fft_matrix * window.unsqueeze(0)
    matrix_real, matrix_imag = fft_matrix.real, fft_matrix.imag
    # print(fft_matrix.shape)
    # plt.plot(matrix_real[2])
    # plt.plot(matrix_imag[2])
    # plt.show()
    return matrix_real, matrix_imag


class ConvSTFT(nn.Module):
    def __init__(self, n_fft, hop_length, trainable=False):
        super().__init__()

        matrix_real, matrix_imag = get_matrix(n_fft)

        self.conv_real = torch.nn.Conv1d(
            1, n_fft // 2 + 1, n_fft, stride=hop_length, padding=n_fft // 2, bias=False
        )
        self.conv_real.weight.data = matrix_real.unsqueeze(1)
        if not trainable:
            self.conv_real.requires_grad_(False)

        self.conv_imag = torch.nn.Conv1d(
            1, n_fft // 2 + 1, n_fft, stride=hop_length, padding=n_fft // 2, bias=False
        )
        self.conv_imag.weight.data = matrix_imag.unsqueeze(1)
        if not trainable:
            self.conv_imag.requires_grad_(False)

    def forward(self, x):
        real = self.conv_real(x)
        imag = self.conv_imag(x)
        return real, imag


class ConvISTFT(nn.Module):
    def __init__(self, n_fft, hop_length, trainable=False):
        super().__init__()

        self.scale = n_fft / hop_length

        matrix_real, matrix_imag = get_matrix(n_fft)
        matrix_inverse_real = torch.linalg.pinv(matrix_real).T
        matrix_inverse_imag = torch.linalg.pinv(matrix_imag).T

        self.convtrans_real = torch.nn.ConvTranspose1d(
            n_fft // 2 + 1, 1, n_fft, stride=hop_length, padding=n_fft // 2, bias=False
        )
        self.convtrans_real.weight.data = matrix_inverse_real.unsqueeze(1)
        if not trainable:
            self.convtrans_real.requires_grad_(False)

        self.convtrans_imag = torch.nn.ConvTranspose1d(
            n_fft // 2 + 1, 1, n_fft, stride=hop_length, padding=n_fft // 2, bias=False
        )
        self.convtrans_imag.weight.data = matrix_inverse_imag.unsqueeze(1)
        if not trainable:
            self.convtrans_imag.requires_grad_(False)

    def forward(self, real, imag):
        wav = (self.convtrans_real(real) + self.convtrans_imag(imag)) / self.scale
        return wav


def devisable_padding(x, hop_length):
    if x.shape[-1] % hop_length != 0:
        x = torch.nn.functional.pad(x, (0, hop_length - x.shape[-1] % hop_length))
    return x


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
        real, imag = conv_stft(wav)
        stft_complex = torch.complex(real, imag)
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
        (output_real, output_imaj) = ort_session.run(None, {input_name: wav})
        print(output_real.shape, output_imaj.shape)
        output_mag = np.sqrt(output_real[0] ** 2 + output_imaj[0] ** 2)
        plt.imshow(np.log(output_mag + 1e-6), origin="lower", aspect="auto")
        plt.show()

    def test_istft():
        wav = load_test_wav().unsqueeze(0)
        wav = devisable_padding(wav, hop_length)
        conv_stft = ConvSTFT(n_fft, hop_length)
        (real, imag) = conv_stft(wav)

        istft = ConvISTFT(n_fft, hop_length)
        wav_recon = istft(real, imag)

        torchaudio.save("recon.wav", wav_recon[0], sample_rate)
        torchaudio.save("origin.wav", wav[0], sample_rate)

        print(torch.abs(wav_recon - wav).mean())

    def test_istft_onnx():
        wav = load_test_wav().unsqueeze(0)
        wav = devisable_padding(wav, hop_length)
        conv_stft = ConvSTFT(n_fft, hop_length)
        (real, imag) = conv_stft(wav)

        istft = ConvISTFT(n_fft, hop_length)
        wav_recon = istft(real, imag)

        export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
        onnx_program = torch.onnx.dynamo_export(
            istft, real, imag, export_options=export_options
        )
        onnx_program.save("model_istft.onnx")

        ort_session = ort.InferenceSession("model_istft.onnx")
        input_dict = {
            "l_real_": real.detach().cpu().numpy(),
            "l_imag_": imag.detach().cpu().numpy(),
        }
        ort_output = ort_session.run(None, input_dict)

        print(torch.abs(torch.from_numpy(ort_output[0]) - wav_recon).mean())

    test()
    test_onnx()
    test_istft()
    test_istft_onnx()
