## 第一回 コマンド一覧

### python のインストール

```bash
$ sudo apt install -y python3.8
$ python3.8 --version

Python 3.8.0 と表示されればOKです

```

### python の pip と apt 周りの変更

```bash
$ sudo apt install -y python3-pip
$ python3.8 -m pip install --upgrade pip
```

### pytorch の インストール

```bash
$ python3.8 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

### git の インストール

```bash
$ sudo apt install -y git
```

### トラブルシューティング

> **Warning** vscode の表示が乱れる場合

- GPU アクセラレーションの無効にして vscode を起動
  ```
  $ code --disable-gpu .
  ```
- 常に VSCode の GPU アクセラレーションを無効化する

参考: https://qiita.com/zakkied/items/f0d21c6cbd8e34460253
