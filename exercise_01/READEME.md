## 第一回 コマンド一覧

### python のインストール
```bash
$ sudo apt install -y python3.8
$ sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
$ sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
$ sudo update-alternatives --config python3

「2」を選択して「Enter」
```

### python の pip と apt 周りの変更
```bash
$ sudo apt install -y python3-pip

$ sudo apt remove python3-apt
$ sudo apt autoremove
$ sudo apt autoclean
$ sudo apt install python3-apt

$ python3 -m pip install --upgrade pip
```

### pytorch の インストール
```bash
$ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

### git の インストール
```bash
$ sudo apt install -y git
```

### トラブルシューティング
> **Warning** vscodeの表示が乱れる場合
- GPU アクセラレーションの無効にして vscode を起動
    ```
    $ code --disable-gpu .
    ```
- 常に VSCode の GPU アクセラレーションを無効化する

参考: https://qiita.com/zakkied/items/f0d21c6cbd8e34460253
