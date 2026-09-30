"""Prepare English<->Chinese bilingual chat data in the exact format used for
the WMT24 Chat Shared Task data in this repository.

Two sources are produced:

1. BMELD (REAL data) -- the bilingual Chinese-English chat translation corpus
   of Liang et al. (ACL 2021, https://github.com/XL2248/CPCC). It is built on
   MELD (Friends TV series dialogues) with human post-edited Chinese
   translations. Following the BMELD/BConTrasT convention (and
   `preprocess_ench.py` in the CPCC repo), the utterances of Ross, Joey and
   Rachel are treated as produced by *Chinese* speakers (zh->en, role
   "customer") and all other speakers as *English* speakers (en->zh, role
   "agent"), mirroring the agent(English)/customer(xx) roles of WMT24 Chat.
   Splits: bmeld_train / bmeld_dev / bmeld_test.

2. synthetic_chat_test (SYNTHETIC data) -- a small customer-support style set
   written for pipeline testing ONLY. It is clearly labelled: doc_ids start
   with "SYNTHETIC-", client_id is "SYNTHETIC" and tags are "['synthetic']".
   Its references are hand-written and have NOT been validated by
   professional translators; do not use it to report research results.

Outputs (mirroring the existing wmt24_chat files):
  downloaded_data/<data>_<split>.en-zh.csv       (source_language,target_language,source,reference,doc_id,client_id,sender,tags)
  raw_data/mt/<data>_<split>.zh/test.jsonl        (both directions, conversation order)
  raw_data/mt/<data>_<split>.en-zh/test.jsonl     (en->zh only)
  raw_data/mt/<data>_<split>.zh-en/test.jsonl     (zh->en only)
  paper_results_zh/<data>/<split>.en-zh.csv       (+ "lp" column; input to MBR / contrastive decoding)

Usage:
  python scripts/prepare_zh_data.py [--bmeld_dir /path/to/CPCC/BMELD_data] [--root_dir .]
If --bmeld_dir is not given, the three BMELD csv files are downloaded from
raw.githubusercontent.com.
"""

import argparse
import csv
import io
import re
import urllib.request
from pathlib import Path

import pandas as pd

BMELD_URL = "https://raw.githubusercontent.com/XL2248/CPCC/master/BMELD_data/{split}_sent_emo.csv"
# Speakers treated as Chinese speakers in BMELD (see CPCC/preprocess_ench.py)
BMELD_ZH_SPEAKERS = {"Ross", "Joey", "Rachel"}
COLUMNS = [
    "source_language",
    "target_language",
    "source",
    "reference",
    "doc_id",
    "client_id",
    "sender",
    "tags",
]


def detok_en(s: str) -> str:
    """MELD utterances are tokenized ("he's lost it ."); undo the space before
    punctuation only. Nothing else is changed."""
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def read_bmeld(split: str, bmeld_dir) -> list:
    if bmeld_dir is not None:
        raw = Path(bmeld_dir, f"{split}_sent_emo.csv").read_bytes()
    else:
        with urllib.request.urlopen(BMELD_URL.format(split=split)) as r:
            raw = r.read()
    # BMELD csv files are GB18030 encoded
    text = raw.decode("gb18030")
    return list(csv.DictReader(io.StringIO(text)))


def bmeld_to_df(rows: list, split: str) -> pd.DataFrame:
    out = []
    for r in rows:
        en = detok_en(r["Utterance"])
        zh = r["Target"].strip()
        if not en or not zh:
            continue
        doc_id = f"bmeld_{split}_{int(r['Dialogue_ID']):04d}"
        client_id = f"friends_s{int(r['Season']):02d}e{int(r['Episode']):02d}"
        if r["Speaker"] in BMELD_ZH_SPEAKERS:
            out.append(["zh", "en", zh, en, doc_id, client_id, "customer", "[]"])
        else:
            out.append(["en", "zh", en, zh, doc_id, client_id, "agent", "[]"])
    return pd.DataFrame(out, columns=COLUMNS)


# ---------------------------------------------------------------------------
# SYNTHETIC customer-support conversations (labelled; for pipeline tests only)
# Each turn: (sender, text_in_speaker_language, reference_translation)
# agent = English speaker (en->zh), customer = Chinese speaker (zh->en)
# ---------------------------------------------------------------------------
SYNTHETIC_CONVERSATIONS = [
    [
        ("customer", "你好，我昨天买的耳机到现在还没发货。", "Hello, the headphones I bought yesterday still haven't been shipped."),
        ("agent", "Hello NAME-M, thank you for contacting us.", "您好，NAME-M，感谢您联系我们。"),
        ("agent", "I'm sorry to hear about the delay with your order.", "很抱歉听到您的订单延迟了。"),
        ("agent", "Could you please provide me with your order number?", "能请您提供一下订单号吗？"),
        ("customer", "订单号是 NUMBER。", "The order number is NUMBER."),
        ("agent", "Thank you. Please allow me a moment to check it.", "谢谢。请稍等，我查一下。"),
        ("agent", "I can see that it is still being processed in our warehouse.", "我看到它仍在我们的仓库中处理。"),
        ("customer", "那它什么时候能发出？", "So when will it be shipped?"),
        ("agent", "It should be shipped within 24 hours.", "它应该会在24小时内发货。"),
        ("agent", "You will receive a tracking number by email once it leaves the warehouse.", "它一离开仓库，您就会通过电子邮件收到快递单号。"),
        ("customer", "好的，如果明天还没发货怎么办？", "Okay, what if it still hasn't shipped by tomorrow?"),
        ("agent", "In that case, please contact us again and we will escalate it.", "如果是那样，请再次联系我们，我们会升级处理。"),
        ("customer", "明白了，谢谢。", "Understood, thank you."),
        ("agent", "You're welcome! Is there anything else I can help you with?", "不客气！还有什么我可以帮您的吗？"),
        ("customer", "没有了。", "No, that's all."),
        ("agent", "Have a nice day!", "祝您有美好的一天！"),
    ],
    [
        ("agent", "Hi, my name is NAME-F. How can I help you today?", "您好，我是NAME-F。今天有什么可以帮您的？"),
        ("customer", "我的账户被锁了，登录不了。", "My account has been locked and I can't log in."),
        ("customer", "我已经试了好几次重置密码。", "I have already tried resetting my password several times."),
        ("agent", "I understand how frustrating that must be.", "我理解这一定让您很困扰。"),
        ("agent", "Did you receive the password reset email?", "您收到重置密码的邮件了吗？"),
        ("customer", "收到了，但是链接打不开。", "Yes, I received it, but the link doesn't open."),
        ("agent", "The link expires after 30 minutes, so it may no longer be valid.", "该链接在30分钟后失效，所以它可能已经无效了。"),
        ("agent", "I have just sent you a new one.", "我刚刚给您发了一个新的。"),
        ("customer", "新的也不行。", "The new one doesn't work either."),
        ("agent", "Could you tell me which browser you are using?", "您能告诉我您使用的是哪种浏览器吗？"),
        ("customer", "手机上的浏览器。", "The browser on my phone."),
        ("agent", "Please try opening it on a computer instead.", "请改用电脑打开它试试。"),
        ("customer", "用电脑可以了！密码已经改好了。", "It works on the computer! The password has been changed."),
        ("agent", "Great! Your account is now unlocked.", "太好了！您的账户现在已经解锁了。"),
        ("customer", "太感谢你了。", "Thank you so much."),
    ],
    [
        ("customer", "我想退掉上周买的那件外套。", "I would like to return the coat I bought last week."),
        ("agent", "Sure, I can help you with the return.", "好的，我可以帮您办理退货。"),
        ("agent", "May I know the reason for returning it?", "请问退货的原因是什么？"),
        ("customer", "尺码太小了，穿不上。", "The size is too small, I can't wear it."),
        ("agent", "Would you like to exchange it for a larger size instead?", "您想换一个大一点的尺码吗？"),
        ("customer", "不用了，直接退款吧。", "No need, just refund it."),
        ("agent", "No problem. The refund will go back to your original payment method.", "没问题。退款将退回到您原来的支付方式。"),
        ("customer", "大概要多久？", "How long will it take roughly?"),
        ("agent", "It usually takes 5 to 7 business days after we receive the item.", "我们收到商品后，通常需要5到7个工作日。"),
        ("customer", "运费谁出？", "Who pays for the shipping?"),
        ("agent", "Return shipping is free for this item.", "这件商品的退货运费是免费的。"),
        ("agent", "I will send you a prepaid return label by email.", "我会通过电子邮件给您发送一张预付退货标签。"),
        ("customer", "好的，收到了。", "Okay, I got it."),
        ("agent", "Please print it and attach it to the package.", "请把它打印出来贴在包裹上。"),
    ],
    [
        ("agent", "Thank you for waiting. I have checked your subscription.", "感谢您的耐心等待。我已经查看了您的订阅。"),
        ("customer", "为什么这个月扣了两次钱？", "Why was I charged twice this month?"),
        ("agent", "It looks like the payment was processed twice by mistake.", "看起来这笔付款被错误地处理了两次。"),
        ("agent", "I apologize for the inconvenience.", "给您带来不便，我深表歉意。"),
        ("customer", "那多扣的钱能退吗？", "Can the extra charge be refunded then?"),
        ("agent", "Yes, I have already requested a refund for the duplicate charge.", "可以，我已经为重复扣款申请了退款。"),
        ("customer", "退到哪里？", "Where will it be refunded to?"),
        ("agent", "It will be returned to the same credit card.", "它会退回到同一张信用卡上。"),
        ("customer", "那张卡我已经注销了。", "I have already cancelled that card."),
        ("agent", "In that case, the bank will usually transfer it to your new card or account.", "如果是那样，银行通常会把它转到您的新卡或账户上。"),
        ("agent", "If it doesn't arrive within 10 days, please let us know.", "如果10天内没有到账，请告诉我们。"),
        ("customer", "好，我会留意的。", "Okay, I'll keep an eye on it."),
        ("agent", "Is there anything else I can do for you?", "还有什么我可以为您做的吗？"),
        ("customer", "暂时没有了，谢谢。", "Not for now, thanks."),
    ],
]


def synthetic_df() -> pd.DataFrame:
    out = []
    for i, convo in enumerate(SYNTHETIC_CONVERSATIONS):
        doc_id = f"SYNTHETIC-enzh-{i:04d}"
        for sender, text, ref in convo:
            if sender == "agent":
                out.append(["en", "zh", text, ref, doc_id, "SYNTHETIC", sender, "['synthetic']"])
            else:
                out.append(["zh", "en", text, ref, doc_id, "SYNTHETIC", sender, "['synthetic']"])
    return pd.DataFrame(out, columns=COLUMNS)


def write_dataset(df: pd.DataFrame, data_name: str, split: str, root: Path) -> None:
    dataset_name = f"{data_name}_{split}"
    download_dir = root / "downloaded_data"
    download_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(download_dir / f"{dataset_name}.en-zh.csv", index=False)

    # same processing as notebooks/preprocess_data_wmt24_test.ipynb
    rdf = df.rename(columns={"source": "src", "reference": "ref"}).copy()
    rdf["lp"] = rdf["source_language"] + "-" + rdf["target_language"]
    raw_dir = root / "raw_data" / "mt"
    for name, sub in [
        (f"{dataset_name}.zh", rdf),
        (f"{dataset_name}.en-zh", rdf[rdf["source_language"] == "en"]),
        (f"{dataset_name}.zh-en", rdf[rdf["target_language"] == "en"]),
    ]:
        d = raw_dir / name
        d.mkdir(parents=True, exist_ok=True)
        sub.reset_index(drop=True).to_json(d / "test.jsonl", orient="records", lines=True, force_ascii=False)

    # paper_results-style csv (source/reference + lp) used by MBR and contrastive decoding
    pr = df.copy()
    pr["lp"] = pr["source_language"] + "-" + pr["target_language"]
    pr_dir = root / "paper_results_zh" / data_name
    pr_dir.mkdir(parents=True, exist_ok=True)
    pr.to_csv(pr_dir / f"{split}.en-zh.csv", index=False)
    n_en = (df["source_language"] == "en").sum()
    print(f"{dataset_name}: {len(df)} segments ({n_en} en->zh, {len(df) - n_en} zh->en), {df['doc_id'].nunique()} conversations")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bmeld_dir", default=None, help="Local CPCC/BMELD_data folder (otherwise downloaded)")
    p.add_argument("--root_dir", default=".")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--skip_bmeld", action="store_true")
    p.add_argument("--skip_synthetic", action="store_true")
    args = p.parse_args()
    root = Path(args.root_dir)
    if not args.skip_bmeld:
        for split in args.splits:
            write_dataset(bmeld_to_df(read_bmeld(split, args.bmeld_dir), split), "bmeld", split, root)
    if not args.skip_synthetic:
        write_dataset(synthetic_df(), "synthetic_chat", "test", root)


if __name__ == "__main__":
    main()
