#!/bin/bash
# English<->Chinese chat data (BMELD, Liang et al. 2021) + synthetic test set, converted to the
# WMT24 Chat layout (downloaded_data/, raw_data/mt/, paper_results_zh/), then all prompt conditions.
set -e
python scripts/prepare_zh_data.py --root_dir .
python scripts/make_instructions.py --datasets bmeld_train --conditions no_context full_context no_context_empty_sys full_context_empty_sys
python scripts/make_instructions.py --datasets bmeld_test bmeld_dev synthetic_chat_test
