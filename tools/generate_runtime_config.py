#!/usr/bin/env python3
import json
from pathlib import Path
c=json.loads(Path('configs/runtime-v0.json').read_text())
text='// Generated from configs/runtime-v0.json; do not edit.\n' + '\n'.join(f'pub const {k.upper()}: i64 = {int(v)};' for k,v in c.items() if isinstance(v,int))+'\n'
Path('service/generated/runtime_config.rs').write_text(text)
