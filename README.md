# tokuder-neurodec

> [!WARNING]
> **これは実験的な実装です。** 標準のモデルでも約 440 万パラメータ（重み 17 MiB）のニューラルネットで、マイコンでの動作には向きません。

TOKUDER（ARIB STD-T99 デジタル小電力コミュニティ無線（DRCR）の音声コーデック）の TCH 音声フレームを、音声に戻すニューラルデコーダーです。
TOKUDER の仕様は公開されていないため、実機で送信したクリーン音声と、受信したフレームの組から、デコーダーを学習しました。

## 中身

| 部分 | 内容 |
|---|---|
| 誤り訂正（FEC） | 受信フレームだけから解読した TCH の誤り訂正（2 × 拡大 Golay [24,12] ＋ 無保護 24 bit、3 bit 周期のインターリーブ）。軟判定の最尤復号と硬判定復号（[仕様と根拠](docs/fec.md)） |
| デコーダー | 48 bit/20 ms → 8 kHz 音声。ビットトークンのフロントエンド（メル・F0・有声度・帯域非周期性を予測）＋ iSTFT ボコーダー（Vocos 型、調波＋雑音の励振） |
| 学習済みモデル | `models/<名前>.pt`（重み）と `models/<名前>.json`（構造・学習条件・評価値）。[下の「モデル」](#モデル)を参照 |
| 入力形式 | 9 バイト/フレームのバイナリ（＋任意で LLR）、16 進テキスト、受信ログのディレクトリ（rf.jsonl、rf.llr） |
| 試聴用の例 | `examples/qwen3-tts/`: 元の音声と、それを実機で送信・SDR で受信し、このデコーダーで復号した音声（[下の「試聴」](#試聴)） |
| サンプル | `samples/`: 実機から受信してキャプチャした、そのままの TCH 音声フレームと LLR（VCTK の未知話者 4 人、各約 9 秒。[説明](samples/README.md)） |

モデルの学習には [neurodec](https://github.com/tallcat4/neurodec) を使いました。このリポジトリには推論に必要な部分だけを入れています。パラメータ名は neurodec と同じで、neurodec のチェックポイントの重み（`G`）をそのまま読み込めます。

## モデル

| | `tokuder48_causal_v1`（標準） | `tokuder48_v1` |
|---|---|---|
| 構造 | 因果的。フレーム t の復号に使うのは、フレーム t+1 までのビット | 非因果的。約 0.5 秒（25 フレーム）先のフレームまで使う |
| 遅延 | 20 ms（1 フレーム）の先読み ＋ iSTFT の重ね合わせ 32 ms | 約 0.5 秒 |
| パラメータ数（重み） | 約 440 万（17 MiB） | 約 950 万（36 MiB） |
| CPU 1 スレッドでの速さ | 音声の長さの約 1/120 | 約 1/90 |

`tokuder2wav` は標準で `tokuder48_causal_v1` を使います。`--model models/tokuder48_v1.pt` で切り替えられます。
因果的なモデルは、未来のフレームを 1 つしか使わないので、受信しながら逐次に復号する用途に向いています（このリポジトリの `decode` は、まとめて復号する実装です）。

## 品質

未知話者の 219 発話（VCTK・JVS、受信フレームの誤り率 ≒ 0）での評価です。

| 指標 | `tokuder48_causal_v1` | `tokuder48_v1` |
|---|---|---|
| PESQ（NB） | 2.53 | 2.46 |
| STOI | 0.80 | 0.80 |
| log-mel 距離 | 4.76 dB | 4.93 dB |
| F0 誤差（RMSE） | 88 cent | 84 cent |

- **ビット誤りへの強さ**: 受信フレームにビット誤り 1 %、3 % を加えると、`tokuder48_causal_v1` の PESQ は 2.51、2.44 でした。
- **比べている相手**: 正解として学習・評価に使ったのは、無線機に入れる前のクリーン音声です。実機のデコーダーが出す音とは比べていません。
- **得意・不得意**:
  - 学習データは音声だけです。トーンなどの音声以外の信号は再現しません。規格書の 1031 Hz 試験データも、トーンとしては復号されません。
  - 低い声（男性）は、高い声よりやや苦手です。

## 試聴

合成音声（Qwen3-TTS で生成、CC0）を実機の無線機で送信し、SDR で受信したフレームを、このデコーダーで復号しました。

| 声 | 読み上げた文 | 元の音声 | 実機で送信 → SDR で受信 → 本デコーダーで復号 |
|---|---|---|---|
| 女性 | このソフトウェアは、すべての人のために、常に自由であり続けます。 | [original.wav](https://github.com/tallcat4/tokuder-neurodec/raw/main/examples/qwen3-tts/female/original.wav) | [decoded.wav](https://github.com/tallcat4/tokuder-neurodec/raw/main/examples/qwen3-tts/female/decoded.wav)（標準）、[tokuder48_v1](https://github.com/tallcat4/tokuder-neurodec/raw/main/examples/qwen3-tts/female/decoded_tokuder48_v1.wav) |
| 男性 | GPL-3.0は、研究、改変、共有する自由を保証するオープンなライセンスです。 | [original.wav](https://github.com/tallcat4/tokuder-neurodec/raw/main/examples/qwen3-tts/male/original.wav) | [decoded.wav](https://github.com/tallcat4/tokuder-neurodec/raw/main/examples/qwen3-tts/male/decoded.wav)（標準）、[tokuder48_v1](https://github.com/tallcat4/tokuder-neurodec/raw/main/examples/qwen3-tts/male/decoded_tokuder48_v1.wav) |

- **受信したフレーム**: 受信したそのままのフレームと LLR も同梱しています。手元で復号し直せます（[examples/qwen3-tts](examples/qwen3-tts/README.md)）。
- **元の音声との比較**: 元の音声を 8 kHz に落として比べると、PESQ は標準のモデルで女性 2.79・男性 2.67（`tokuder48_v1` は 2.81・2.57）、STOI は女性 0.86・男性 0.88（同 0.85・0.87）です。

## 動作環境

- Python 3.10 以上、PyTorch 2.1 以上、numpy、soundfile
- CPU でも動きます。デスクトップ CPU の 1 スレッドで、音声の長さの約 1/100 の時間で復号できます（[モデル](#モデル)）。CUDA があれば自動で使います。

## インストール

```bash
pip install -e .        # tokuder2wav コマンドが入ります（モデルはリポジトリの models/ を使います）
pip install -e .[test] && pytest
```

## 使い方

```bash
tokuder2wav samples/vctk_p301_female.bin out.wav --llr samples/vctk_p301_female.llr   # 同梱のサンプル
tokuder2wav frames.bin out.wav                    # 9 バイト/20 ms の受信フレーム（72 bit、送信順、MSB 先頭）
tokuder2wav frames.bin out.wav --llr frames.llr   # int8 の LLR [フレーム数, 72]（正 = 1、全 0 = 消失）で軟判定
tokuder2wav frames.hex out.wav                    # 1 行 1 フレームの 16 進 18 桁（空行か "-" は消失フレーム）
tokuder2wav path/to/rx_log out.wav                # 受信ログのディレクトリ（rf.jsonl、rf.llr）
tokuder2wav frames.bin out.wav --slot 2           # 先頭フレームが RF フレーム（80 ms）内の第 2 スロットのとき
python -m tokuder_neurodec ...                    # インストールせずに実行
```

### 入力についての注意

- **フレームの順序**: フレームは時間順に並べてください。受信できなかったフレームは、消失フレームとして同じ位置に入れてください。消失フレームは直前のフレームで置き換えます（学習時と同じ処理です）。
- **スロット位置**: デコーダーは、各フレームが 80 ms の RF フレーム内のどのスロット（0〜3）にあるかを使います。受信ログのディレクトリなら自動で合わせます。それ以外の入力で、先頭が RF フレームの切れ目でない場合は、`--slot` を指定してください。
- **出力の再現性**: 出力の波形サンプルは、励振の雑音の乱数（`--seed`）によってわずかに変わります。

### Python から

```python
from tokuder_neurodec import load, decode, frames
bits, llr, erased = frames.read_t99("rx_log")
info, _ = frames.to_info48(bits, llr, erased)      # FEC 復号 → 48 bit/フレーム
wav = decode(load("models/tokuder48_causal_v1.pt"), info)  # float32、8 kHz
```

## 学習データ

- **フレームデータ**: 音声を実機で送信して受信したもので、約 6 時間、学習に使ったのは 5.5 時間です。
- **送信した音声**:
  - [VCTK](https://doi.org/10.7488/ds/2645)（CC BY 4.0）
  - [JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)（個人・非商用の研究目的に限る、再配布不可）
  - [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)（配布元の利用規約による）
- **ボコーダー部分**: クリーン音声で事前学習し、デコーダーの学習中は固定しました。`tokuder48_causal_v1` は上の学習用の送信音声（VCTK・JVS・JSUT）、`tokuder48_v1` は JVS・JSUT を使っています。
- **サンプルに使った音声**: `samples/` のフレームには、再配布できる VCTK の音声だけを使っています。

## ライセンス

- コード: GPL-3.0-or-later（GNU GPL バージョン 3 またはそれ以降の任意のバージョン。[LICENSE](LICENSE)）。
- `samples/` のフレーム: CC BY 4.0（元の音声は VCTK。[samples/README.md](samples/README.md)）。
- `examples/qwen3-tts/` の音声とフレーム: CC0 1.0（[examples/qwen3-tts/README.md](examples/qwen3-tts/README.md)）。
- 学習済みの重み: 下の注記を参照。

<sub>学習データの一部には配布元の利用規約があり、JVS は非商用・研究目的の利用に限られ、JSUT にも配布元の利用規約があります。法的な判断を断言することはできませんが、日本の著作権法第 30 条の 4 第 2 号（情報解析のための利用。著作隣接権にも第 102 条第 1 項で準用）により、これらのコーパスをこのモデルの学習に用いることは著作権法上許されると考えられます。また、学習済みの重みは各コーパスの音声を含まず復元もしないため、重みの配布はコーパスの再配布には当たらないと考えられます。ただし、コーパスの利用規約の解釈や日本国外での扱いについては、利用者ご自身で確認してください。</sub>

<sub>TOKUDER はアイコム株式会社の登録商標です。このリポジトリはアイコム株式会社とは関係がなく、同社の承認・関与を受けたものではありません。「TOKUDER」の名称は、ARIB STD-T99 の音声フレームとの互換性を示す目的でのみ使用しています。</sub>
