# 試聴用の例: 合成音声を実機で送信して、このデコーダーで復号

| ファイル | 内容 |
|---|---|
| `<声>/original.wav` | 送信した元の音声（24 kHz） |
| `<声>/frames.bin`、`frames.llr` | 実機の無線機から送信した電波を SDR で受信してキャプチャした、そのままの TCH 音声フレームと、受信機が出した LLR（形式は [samples/README.md](../../samples/README.md) と同じ） |
| `<声>/decoded.wav` | 上のフレームを、このリポジトリのデコーダーで復号した音声（8 kHz） |
| `<声>/text.txt` | 読み上げた文 |
| `generation.json` | 音声の生成条件（モデル、リビジョン、声の指示文、シード） |

- **元の音声**: [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)（Apache-2.0）で、文章から新しく生成したものです。
  - 実在の人物の声の複製（クローン）や、外部の参照音声は使っていません。
  - ピッチの加工や継ぎ接ぎもしていません。
- **フレームの中身**: 入っているのは TCH（音声）フレームだけです。制御チャネルは含みません。
  - 消失フレームはありません。
  - 全フレームに受信機自身の LLR が付いています（値を補ったフレームはありません）。
- **復号の再現**: `decoded.wav` は次のコマンドで作りました。

  ```bash
  tokuder2wav examples/qwen3-tts/female/frames.bin decoded.wav --llr examples/qwen3-tts/female/frames.llr
  ```

## ライセンス

このディレクトリの音声とフレームは、[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) で提供します。権利を有する範囲で、すべての権利を放棄します。
この宣言は、音声を生成したモデルや第三者の権利を再許諾するものではありません。
