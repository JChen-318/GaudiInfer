/* GaudiInfer Studio & Monitor Frontend (Zero-dependency Apple UI, Bilingual ZH/EN) */
const ICONS = {
  pause: '<path d="M9 5v14M15 5v14"/>',
  play: '<path d="m7 4 13 8-13 8z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
};
const ico = (n, s = 14) => `<svg class="ico" width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">${ICONS[n] || ''}</svg>`;
document.querySelectorAll('[data-ic]').forEach(el => { el.outerHTML = ico(el.dataset.ic, +(el.dataset.s || 14)); });

/* ==================== i18n Dictionary (ZH / EN) ==================== */
const I18N = {
  zh: {
    nav_llm: 'LLM 对话', nav_qwen21: 'Qwen 2.1 生图', nav_h3i: 'H3 图像', nav_h3: 'H3 视频',
    nav_gallery: '作品画廊', nav_monitor: '硬件监控',
    hw_mini_tip: '点击查看完整硬件监控', theme_tip: '切换深浅色主题', pause_tip: '暂停 / 继续',
    connecting: '连接中…', disconnected: '连接断开', switching: '切换中…', running: '运行中', card_idle: '卡空闲',
    elapsed: '耗时', view_logs: '查看日志', start: '启动', stop: '停止', loading: '加载中…',
    clear: '清空', refresh: '刷新', send: '发送', close_esc: '关闭 (Esc)', download: '下载', del: '删',
    need_llm_t: '对话模型当前未加载',
    need_llm_s: '单卡 Gaudi2 每次常驻一个模型，点击右侧按钮一键切换（/dev/shm 加载极速就绪）',
    launch_llm: '一键启动 Qwen3.8-27B', launch_ink: '一键启动 Inkling-2-Mini',
    need_q21_t: 'Qwen-Image-2.1-Turbo 当前未加载',
    need_q21_s: '1024² 文生图 2.3s、1–4 图编辑 4.2s（占用 22.6 GiB HBM）',
    launch_q21: '一键启动 Qwen 2.1 生图',
    need_h3i_t: 'MiniMax-H3-Image 当前未加载',
    need_h3i_s: '33.1B 原生 2K（4.6s）& 4K 16MP（22s）摄影级画质，支持 1–16 张参考图编辑',
    launch_h3i: '一键启动 H3 图像',
    need_h3_t: 'MiniMax-H3 视频引擎当前未加载',
    need_h3_s: '支持文生音视频（T2VA）、首帧生视频（I2VA）与首尾帧过渡（FL2VA），同步生成双声道音频',
    launch_h3: '一键启动 H3 视频',

    chat_cfg_t: '对话设置', chat_cfg_s: 'Qwen3.8-27B / Inkling-2-Mini · 视觉多模态', chat_new: '新对话',
    chat_think: '开启深度思考 (Thinking)', chat_temp: '温度 Temp', chat_max: '最大输出',
    chat_sys: '系统提示词 (可选)', chat_sys_ph: '例如：你是一个专业视觉与代码助手，回答简洁直接。',
    quick_presets: '快捷提示', chat_add_img: '📎 图片', chat_add_img_tip: '上传图片给 27B 分析',
    chat_in_ph: '输入消息…（Enter 发送，Shift+Enter 换行，支持 Ctrl+V 粘贴图片）',
    chat_empty_t: 'Qwen3.8-27B 多模态对话工作台',
    chat_empty_s: '支持文本、代码、数学推理与图片理解（可直接粘贴 Ctrl+V 或点击下方附加图片），也可以把生图结果一键发来分析。',
    chat_new_t: '新对话已就绪', chat_new_s: '输入问题或粘贴图片即可与 Qwen3.8-27B 对话。',
    role_user: '你', think_title: '思考过程', chars_unit: '字', wait_resp: '等待响应…',
    streaming_meta: '流式输出中', think_only_msg: '（内容已全部输出至思考过程，可点开上方折叠栏查看）',
    img_only_prompt: '请详细描述这张图片。', img_analysis_lbl: '（图片分析）',
    req_fail: '请求失败', ensure_27b: '请确认 Qwen3.8-27B 引擎已启动', failed: '失败',

    q21_sub: '8 步极速文生图 & 1–4 参考图编辑',
    q21_prompt_lbl: '提示词 (中英文皆可，文字排版能力极强)',
    q21_prompt_ph: '描述你想生成的画面，或添加参考图后描述如何修改…',
    q21_ref_lbl: '参考图（可选，最多 4 张 · 放入即自动切换为图像编辑）',
    drop_title: '点击选择 / 拖拽图片到此处',
    q21_drop_sub: '无参考图 = 文生图；1–4 张参考图 = 保持主体编辑',
    res_presets: '分辨率预设', width_lbl: '宽 W', height_lbl: '高 H', steps_lbl: '步数',
    seed_lbl: '随机种子 Seed', rand_seed: '🎲 随机', q21_run: '生成图像',
    q21_stage_t: 'Qwen-Image-2.1-Turbo 画布', q21_stage_s: '点击左侧「生成图像」，1024² 约 2.3 秒直出',
    mode_t2i: '文生图 T2I', mode_edit_n: n => `图像编辑 (${n} 图)`, mode_h3i_edit_n: n => `多图编辑 (${n} 图)`,
    pic_tag: i => `图${i}`,

    h3i_sub: '6 步 PDD · 原生 2K / 4K 16MP · 最多 16 图编辑',
    h3i_prompt_lbl: '提示词（编辑时可用 Picture 1 / Picture 2 指代参考图）',
    h3i_prompt_ph: '描述你想生成的 2K/4K 画面，或结合 Picture 1 描述精准编辑…',
    h3i_ref_lbl: '参考图（可选，最多 16 张 · 按顺序编号为 Picture 1…16）',
    h3i_drop_sub: '支持 1–16 张多图融合或单图精准光照/材质/人物编辑',
    h3i_run: '生成 2K / 4K 图像',
    h3i_stage_t: 'MiniMax-H3-Image 高清画布', h3i_stage_s: '支持原生 2K（4.6s）到 4K 16MP（22s）直出，无需放大器',

    h3_title: 'MiniMax-H3 音视频工作台', h3_sub: 'Direct MME + SRAM · 同步 32kHz 立体声',
    h3_prompt_lbl: '视频与声音描述 (Prompt)',
    h3_prompt_ph: '描述画面运动、镜头语言以及希望听到的环境音/人声/配乐…',
    h3_first_lbl: '首帧图 (可选)', h3_last_lbl: '尾帧图 (可选)',
    h3_first_ph: '点击/拖入首帧', h3_last_ph: '点击/拖入尾帧',
    h3_lora_lbl: '加速 LoRA 预设（单请求热切换）',
    h3_sz_480h: '480p 横屏 (832×480)', h3_sz_480v: '480p 竖屏 (480×832)',
    h3_sz_768c: '768p 电影 (1344×768)', h3_sz_768v: '768p 竖屏 (768×1280)',
    duration_lbl: '时长 Duration', h3_run: '生成音视频 MP4',
    h3_stage_t: 'MiniMax-H3 音视频放映厅', h3_stage_s: '支持 480p / 768p、3–15 秒、同步双声道音频，成片自动存入作品画廊',
    mode_t2va: '文生音视频 T2VA', mode_i2va: '图生视频 I2VA', mode_fl2va: '首尾帧 FL2VA',
    lora_v12: '8步 Turbo v1.2（推荐 · 画质最佳）',
    lora_v10: '4步 Turbo v1.0（极速出片）',
    lora_none: '无 LoRA（Base 50 步原版）',

    act_q21: '✏️ 发给 Qwen 2.1 编辑', act_h3i: '🖼️ 发给 H3-Image 编辑',
    act_h3v: '🎬 设为 H3 视频首帧', act_llm: '💬 发给 27B 看图', act_dl_png: '⬇️ 下载 PNG', act_dl_mp4: '⬇️ 下载 MP4',
    gal_act_h3i: 'H3编辑', gal_act_q21: '2.1编辑', gal_act_vid: '做视频', gal_act_llm: '27B看图',

    gal_title: '作品画廊', gal_sub: '所有生成的 2K/4K 图像与 H3 音视频均持久化保存，可一键转发至任意工作台继续编辑或生成视频',
    gal_all: '全部', gal_empty_t: '画廊暂无作品', gal_empty_s: '在 Qwen 2.1、H3 图像或 H3 视频工作台生成后会自动归档到这里',
    confirm_del: '确认删除该作品？', no_prompt: '无提示词',

    mon_title: 'Gaudi2 实时监控', status_lbl: '状态', updated_lbl: '更新于',
    kpi_avg: '均值', kpi_peak: '极值', kpi_spec: '规格', kpi_detail: '详情',
    kpi_sub: '副指标', kpi_hit: '命中/编码', kpi_stat: '统计', kpi_queue: '轮数/排队',
    server_side: '服务端', hw_card_t: 'Gaudi2 加速器',
    ring_util: 'AIP 利用率', ring_pow: '整卡功耗', ring_temp: 'AIP 温度', ring_clk: 'SoC 频率',
    spec_hbm: 'HBM2E 已用', spec_pcie: 'PCIe 链路', spec_drv: '驱动版本', spec_rst: '硬重置',
    cap_title: '资源与显存占用', host_ram: '宿主机内存',
    tot_req: '累计请求/任务', tot_cpu: '宿主机 CPU', tot_shm: '/dev/shm 缓存', tot_gal: '画廊作品数',
    perf_title: '实时吞吐曲线', perf_sub: '近 5 分钟每秒实时采样', perf_live: '实时', kv_title: 'HBM 与缓存趋势',
    tab_reqs: '请求 / 任务明细', tab_logs: '引擎实时日志', tab_info: '系统信息',
    log_filter_ph: '搜索日志…', auto_scroll: '自动滚动', no_logs: '暂无日志输出',
    info_hw: 'Intel Gaudi2 加速器', info_eng: '当前活跃推理引擎', info_sys: '宿主机与存储',

    // Dynamic monitor & status labels
    idle: '空闲', online: '在线', computing: '计算中', normal: '正常', warm: '偏高', hot: '过热',
    peak: '峰值', limit: '上限', unavail: '不可用', not_loaded: '未加载模型', not_running: '未运行',
    waiting_data: '等待采样数据…', total_t: '总耗时', encode_t: '编码', frames_u: '帧',
    eng_hints: {
      llm: ['~33 GiB + 动态 KV', '~180–210 tok/s · 看图 + 思考'],
      inkling: ['~28 GiB + 动态 KV', '~130–200 tok/s · 看图 + 思考 · 1M'],
      qwen21: ['22.6 GiB', '1024² 约 2.3s · 编辑约 4.2s'],
      h3_image: ['56.9 GiB', '2K 约 4.6s · 4K 16MP 约 22s'],
      h3: ['~62 GiB', '480p/768p 5–15s 同步双声道'],
    },
    t_enter_prompt: '请输入提示词', t_enter_vprompt: '请输入视频提示词',
    t_switching: id => `正在切换至 ${id}，请稍候…`,
    t_stopping: '正在停止当前引擎并释放显存…',
    t_sent_q21: '已发送至 Qwen 2.1 编辑参考图',
    t_sent_h3i: '已发送至 H3-Image 编辑（Picture 1）',
    t_sent_h3v: '已设为 MiniMax-H3 视频首帧',
    t_sent_llm: '已附加图片至 27B 对话框',
    t_max_q21: 'Qwen 2.1 最多支持 4 张参考图',
    t_max_h3i: 'H3-Image 最多支持 16 张参考图',
    t_gen_done: s => `生成完成 · ${s}s`,
    t_hd_done: s => `高清图生成完成 · ${s}s`,
    t_vid_done: s => `视频生成完成 · ${s}s`,
    t_gen_fail: m => `生成失败：${m}`,
    t_vid_fail: m => `视频生成失败：${m}`,
    t_submitting_vid: '提交任务至 MiniMax-H3…',
    t_q21_busy: s => `Qwen 2.1 推理中… ${s}s`,
    t_h3i_busy: s => `MiniMax-H3-Image 渲染中… ${s}s`,
    t_h3_busy: (st, s) => `MiniMax-H3 生成中 (${st})… ${s}s`,
    t_h3_busy_sub: jid => `任务 ID: ${jid} · 可在右上角实时查看 Gaudi2 算力负载`,

    chart_power: '功耗', chart_util: 'AIP 利用率', chart_temp: '温度',
    chart_hbm: 'HBM 已用', chart_live_dec: '实时 Decode', chart_live_pre: '实时 Prefill',
    chart_spec: '投机接受', chart_hit: '前缀命中率',
    presets_chat: [
      ['投机解码', '用三句话解释投机解码（Speculative Decoding）的原理与优势。'],
      ['看图分析', '详细描述这张图片的构图、光线、主体和文字细节。'],
      ['写生图提示词', '帮我写一段高质量英文提示词（用于 MiniMax-H3-Image 生成 4K 电影质感照片），主题是：雨夜东京霓虹街头拉面馆。'],
      ['写代码', '用 Python 写一个支持 TTL 的线程安全 LRU Cache，给出完整可运行代码。'],
    ],
    presets_q21: [
      ['中文咖啡海报', '一张极简高级商业摄影海报，清晨暖阳照进日式原木咖啡馆，桌上放着一杯拿铁拉花和可颂面包，墙面海报清晰写着“早安咖啡”，浅景深，柔和电影光影'],
      ['樱花茶庭', 'Photorealistic architectural photograph of a serene Kyoto cedar tea pavilion beside a koi pond surrounded by blooming pink cherry blossoms, morning mist, soft golden sunlight, Leica 35mm lens.'],
      ['转冬雪夜景(编辑)', '保持原图的建筑、构图和主体完全不变，将季节改为暮色大雪纷飞的冬夜，屋檐积雪，暖黄色纸灯笼亮起，电影质感。'],
    ],
    presets_h3i: [
      ['京都茶室缘侧', 'A photorealistic fine-art photograph of a sunlit Kyoto wooden tea house veranda overlooking a lush moss garden with a small stone koi pond, a handmade ceramic matcha bowl and bamboo whisk resting on a tatami mat, delicate golden afternoon sunlight filtering through maple leaves, ultra-sharp natural lens texture.'],
      ['农庄餐桌俯拍', 'A photorealistic overhead food photograph capturing a rustic farm-to-table feast spread across a heavily weathered dark wooden table: freshly baked golden sourdough loaf in the center, hammered copper bowl of colorful seasonal vegetables, artisan cheeses on a dark board, heirloom tomatoes, red wine glass, warm dappled morning sunlight.'],
      ['Picture 1 转雪夜灯笼', 'Keep the exact composition, objects, and layout of Picture 1, but transform the lighting into serene snowy winter dusk with soft fresh snow covering the outdoor surfaces and warm glowing paper lantern light.'],
    ],
    presets_h3: [
      ['秋雨茶室(声画)', 'A cinematic slow tracking shot of a tranquil Japanese wooden tea house in autumn rain, warm golden paper lanterns glowing softly, maple leaves gently drifting down into a koi pond, soothing sound of gentle rain drops and distant wind chimes.'],
      ['海浪礁石日落', 'Close-up cinematic shot of waves crashing against volcanic black rocks at golden hour sunset, sea spray glistening in warm amber light, realistic ocean surf and seagull ambient audio.'],
      ['首帧自然推镜', 'The camera slowly pushes forward from the first frame, gentle wind rustling the leaves and warm ambient soundscape.'],
    ],
  },
  en: {
    nav_llm: 'LLM Chat', nav_qwen21: 'Qwen 2.1', nav_h3i: 'H3 Image', nav_h3: 'H3 Video',
    nav_gallery: 'Gallery', nav_monitor: 'Monitor',
    hw_mini_tip: 'Open full hardware monitor', theme_tip: 'Toggle Light / Dark theme', pause_tip: 'Pause / Resume',
    connecting: 'Connecting…', disconnected: 'Disconnected', switching: 'Switching…', running: 'Running', card_idle: 'Idle',
    elapsed: 'Elapsed', view_logs: 'Logs', start: 'Start', stop: 'Stop', loading: 'Loading…',
    clear: 'Clear', refresh: 'Refresh', send: 'Send', close_esc: 'Close (Esc)', download: 'Download', del: 'Del',
    need_llm_t: 'No chat model is loaded',
    need_llm_s: 'Single Gaudi2 hosts one engine at a time. Click right to switch (fast load from /dev/shm).',
    launch_llm: 'Launch Qwen3.8-27B', launch_ink: 'Launch Inkling-2-Mini',
    need_q21_t: 'Qwen-Image-2.1-Turbo is not loaded',
    need_q21_s: '1024² T2I in ~2.3s, 1–4 ref editing in ~4.2s (22.6 GiB HBM).',
    launch_q21: 'Launch Qwen 2.1 Image',
    need_h3i_t: 'MiniMax-H3-Image is not loaded',
    need_h3i_s: '33.1B native 2K (~4.6s) & 4K 16MP (~22s) studio quality with 1–16 ref editing.',
    launch_h3i: 'Launch H3 Image',
    need_h3_t: 'MiniMax-H3 Video is not loaded',
    need_h3_s: 'Text-to-Video+Audio (T2VA), First-Frame (I2VA) & First/Last-Frame (FL2VA) with stereo audio.',
    launch_h3: 'Launch H3 Video',

    chat_cfg_t: 'Chat Settings', chat_cfg_s: 'Qwen3.8-27B / Inkling-2-Mini · Multimodal Vision', chat_new: 'New Chat',
    chat_think: 'Enable Thinking', chat_temp: 'Temperature', chat_max: 'Max Tokens',
    chat_sys: 'System Prompt (Optional)', chat_sys_ph: 'e.g., You are a concise vision & coding assistant.',
    quick_presets: 'Quick Presets', chat_add_img: '📎 Image', chat_add_img_tip: 'Attach image(s) for 27B Vision',
    chat_in_ph: 'Type a message… (Enter to send, Shift+Enter newline, Ctrl+V to paste image)',
    chat_empty_t: 'Qwen3.8-27B Multimodal Chat Studio',
    chat_empty_s: 'Supports chat, code, reasoning & vision (paste with Ctrl+V or attach images below, or send outputs from Gallery).',
    chat_new_t: 'New conversation ready', chat_new_s: 'Type a prompt or paste an image to chat with Qwen3.8-27B.',
    role_user: 'You', think_title: 'Thinking Process', chars_unit: 'chars', wait_resp: 'Waiting…',
    streaming_meta: 'Streaming', think_only_msg: '(All output tokens were used in thinking; expand above to read.)',
    img_only_prompt: 'Describe this image in detail.', img_analysis_lbl: '(Image Analysis)',
    req_fail: 'Request failed', ensure_27b: 'Make sure Qwen3.8-27B engine is running', failed: 'Failed',

    q21_sub: '8-step Fast T2I & 1–4 Reference Image Editing',
    q21_prompt_lbl: 'Prompt (EN / ZH supported, strong typography)',
    q21_prompt_ph: 'Describe the scene to generate, or add reference images to edit…',
    q21_ref_lbl: 'Reference Images (Optional, up to 4 · auto-switches to Edit mode)',
    drop_title: 'Click or drag & drop images here',
    q21_drop_sub: '0 refs = Text-to-Image; 1–4 refs = Subject-consistent Editing',
    res_presets: 'Resolution Presets', width_lbl: 'Width', height_lbl: 'Height', steps_lbl: 'Steps',
    seed_lbl: 'Seed', rand_seed: '🎲 Random', q21_run: 'Generate Image',
    q21_stage_t: 'Qwen-Image-2.1-Turbo Canvas', q21_stage_s: 'Click "Generate Image" on the left — 1024² in ~2.3s',
    mode_t2i: 'Text-to-Image', mode_edit_n: n => `Image Edit (${n} ref)`, mode_h3i_edit_n: n => `Multi-Ref Edit (${n})`,
    pic_tag: i => `Img ${i}`,

    h3i_sub: '6-step PDD · Native 2K / 4K 16MP · Up to 16 Refs',
    h3i_prompt_lbl: 'Prompt (Reference images as Picture 1, Picture 2…)',
    h3i_prompt_ph: 'Describe a 2K/4K scene, or reference Picture 1 for precise editing…',
    h3i_ref_lbl: 'Reference Images (Optional, up to 16 · indexed Picture 1…16)',
    h3i_drop_sub: 'Supports 1–16 reference fusion or pixel-locked relighting & editing',
    h3i_run: 'Generate 2K / 4K Image',
    h3i_stage_t: 'MiniMax-H3-Image HD Canvas', h3i_stage_s: 'Native 2K (~4.6s) to 4K 16MP (~22s) direct generation, zero upscaler',

    h3_title: 'MiniMax-H3 Video & Audio Studio', h3_sub: 'Direct MME + SRAM · Synced 32kHz Stereo Audio',
    h3_prompt_lbl: 'Video & Audio Prompt',
    h3_prompt_ph: 'Describe motion, camera movement, and ambient sound / dialogue / music…',
    h3_first_lbl: 'First Frame (Optional)', h3_last_lbl: 'Last Frame (Optional)',
    h3_first_ph: 'Click / drop first frame', h3_last_ph: 'Click / drop last frame',
    h3_lora_lbl: 'Acceleration LoRA (Per-request hot swap)',
    h3_sz_480h: '480p Landscape (832×480)', h3_sz_480v: '480p Portrait (480×832)',
    h3_sz_768c: '768p Cinema (1344×768)', h3_sz_768v: '768p Portrait (768×1280)',
    duration_lbl: 'Duration', h3_run: 'Generate Video + Audio MP4',
    h3_stage_t: 'MiniMax-H3 Screening Room', h3_stage_s: '480p / 768p, 3–15s with synced stereo audio — auto-saved to Gallery',
    mode_t2va: 'Text-to-Video T2VA', mode_i2va: 'Image-to-Video I2VA', mode_fl2va: 'First+Last FL2VA',
    lora_v12: '8-step Turbo v1.2 (Recommended · Best Quality)',
    lora_v10: '4-step Turbo v1.0 (Fastest)',
    lora_none: 'No LoRA (Base 50-step)',

    act_q21: '✏️ Edit in Qwen 2.1', act_h3i: '🖼️ Edit in H3-Image',
    act_h3v: '🎬 Use as H3 First Frame', act_llm: '💬 Analyze in 27B', act_dl_png: '⬇️ Download PNG', act_dl_mp4: '⬇️ Download MP4',
    gal_act_h3i: 'H3 Edit', gal_act_q21: '2.1 Edit', gal_act_vid: 'To Video', gal_act_llm: '27B Vision',

    gal_title: 'Media Gallery', gal_sub: 'All generated 2K/4K images and H3 videos are persisted on disk with one-click cross-model handoff',
    gal_all: 'All', gal_empty_t: 'Gallery is empty', gal_empty_s: 'Images and videos generated in any Studio tab appear here automatically',
    confirm_del: 'Delete this item permanently?', no_prompt: 'No prompt',

    mon_title: 'Gaudi2 Telemetry Monitor', status_lbl: 'Status', updated_lbl: 'Updated',
    kpi_avg: 'Avg', kpi_peak: 'Peak', kpi_spec: 'Spec', kpi_detail: 'Detail',
    kpi_sub: 'Sub', kpi_hit: 'Hit / Enc', kpi_stat: 'Stats', kpi_queue: 'Rounds / Q',
    server_side: 'Server', hw_card_t: 'Gaudi2 Accelerator',
    ring_util: 'AIP Util', ring_pow: 'Total Power', ring_temp: 'AIP Temp', ring_clk: 'SoC Clock',
    spec_hbm: 'HBM2E Used', spec_pcie: 'PCIe Link', spec_drv: 'Driver', spec_rst: 'Resets',
    cap_title: 'Capacity & Memory', host_ram: 'Host RAM',
    tot_req: 'Total Jobs', tot_cpu: 'Host CPU', tot_shm: '/dev/shm Cache', tot_gal: 'Gallery Items',
    perf_title: 'Live Throughput', perf_sub: 'Last 5 min real-time sampling', perf_live: 'Live', kv_title: 'HBM & Cache Trends',
    tab_reqs: 'Requests / Jobs', tab_logs: 'Live Engine Logs', tab_info: 'System Info',
    log_filter_ph: 'Filter logs…', auto_scroll: 'Auto-scroll', no_logs: 'No log output yet',
    info_hw: 'Intel Gaudi2 Accelerator', info_eng: 'Active Inference Engine', info_sys: 'Host & Storage',

    idle: 'Idle', online: 'Online', computing: 'Active', normal: 'Normal', warm: 'Warm', hot: 'Hot',
    peak: 'Max', limit: 'Limit', unavail: 'Unavailable', not_loaded: 'No model loaded', not_running: 'Stopped',
    waiting_data: 'Waiting for samples…', total_t: 'Total', encode_t: 'Encode', frames_u: 'f',
    eng_hints: {
      llm: ['~33 GiB + Dynamic KV', '~180–210 tok/s · Vision + Thinking'],
      inkling: ['~28 GiB + Dynamic KV', '~130–200 tok/s · Vision + Thinking · 1M'],
      qwen21: ['22.6 GiB', '1024² ~2.3s · Edit ~4.2s'],
      h3_image: ['56.9 GiB', '2K ~4.6s · 4K 16MP ~22s'],
      h3: ['~62 GiB', '480p/768p 5–15s Stereo Audio'],
    },
    t_enter_prompt: 'Please enter a prompt', t_enter_vprompt: 'Please enter a video prompt',
    t_switching: id => `Switching to ${id}, please wait…`,
    t_stopping: 'Stopping active engine and releasing HBM…',
    t_sent_q21: 'Sent to Qwen 2.1 reference image',
    t_sent_h3i: 'Sent to H3-Image editing (Picture 1)',
    t_sent_h3v: 'Set as MiniMax-H3 first frame',
    t_sent_llm: 'Attached image to 27B Chat',
    t_max_q21: 'Qwen 2.1 supports up to 4 reference images',
    t_max_h3i: 'H3-Image supports up to 16 reference images',
    t_gen_done: s => `Generated in ${s}s`,
    t_hd_done: s => `HD image ready in ${s}s`,
    t_vid_done: s => `Video ready in ${s}s`,
    t_gen_fail: m => `Generation failed: ${m}`,
    t_vid_fail: m => `Video failed: ${m}`,
    t_submitting_vid: 'Submitting job to MiniMax-H3…',
    t_q21_busy: s => `Qwen 2.1 generating… ${s}s`,
    t_h3i_busy: s => `MiniMax-H3-Image rendering… ${s}s`,
    t_h3_busy: (st, s) => `MiniMax-H3 generating (${st})… ${s}s`,
    t_h3_busy_sub: jid => `Job ID: ${jid} · Live Gaudi2 telemetry in top bar`,

    chart_power: 'Power', chart_util: 'AIP Util', chart_temp: 'Temp',
    chart_hbm: 'HBM Used', chart_live_dec: 'Live Decode', chart_live_pre: 'Live Prefill',
    chart_spec: 'Spec Accept', chart_hit: 'Prefix Hit %',
    presets_chat: [
      ['Speculative Decoding', 'Explain how Speculative Decoding works and why it speeds up LLM inference in 3 sentences.'],
      ['Analyze Image', 'Describe the composition, lighting, subjects, and any visible text in this image in detail.'],
      ['Write 4K Prompt', 'Write a vivid photorealistic prompt for MiniMax-H3-Image depicting a rainy neon Tokyo ramen alley at night.'],
      ['Python LRU Cache', 'Write a clean thread-safe Python LRU Cache with TTL support and a quick usage example.'],
    ],
    presets_q21: [
      ['Minimalist Cafe Poster', 'Minimalist editorial commercial poster of a sunlit Japanese cedar cafe, latte art and croissant on an oak table, wall sign clearly reading "MORNING COFFEE", shallow depth of field, soft cinematic light.'],
      ['Kyoto Sakura Pavilion', 'Photorealistic architectural photograph of a serene Kyoto cedar tea pavilion beside a koi pond surrounded by blooming pink cherry blossoms, morning mist, soft golden sunlight, Leica 35mm lens.'],
      ['Snowy Winter Edit', 'Keep the exact building, composition, and layout unchanged, transform the season into a serene snowy winter dusk with fresh snow on the roof and warm glowing paper lanterns.'],
    ],
    presets_h3i: [
      ['Kyoto Tea Veranda', 'A photorealistic fine-art photograph of a sunlit Kyoto wooden tea house veranda overlooking a lush moss garden with a small stone koi pond, a handmade ceramic matcha bowl and bamboo whisk resting on a tatami mat, delicate golden afternoon sunlight filtering through maple leaves, ultra-sharp natural lens texture.'],
      ['Farm-to-Table Feast', 'A photorealistic overhead food photograph capturing a rustic farm-to-table feast spread across a heavily weathered dark wooden table: freshly baked golden sourdough loaf in the center, hammered copper bowl of colorful seasonal vegetables, artisan cheeses on a dark board, heirloom tomatoes, red wine glass, warm dappled morning sunlight.'],
      ['Picture 1 -> Snowy Dusk', 'Keep the exact composition, objects, and layout of Picture 1, but transform the lighting into serene snowy winter dusk with soft fresh snow covering the outdoor surfaces and warm glowing paper lantern light.'],
    ],
    presets_h3: [
      ['Rainy Tea House', 'A cinematic slow tracking shot of a tranquil Japanese wooden tea house in autumn rain, warm golden paper lanterns glowing softly, maple leaves gently drifting down into a koi pond, soothing sound of gentle rain drops and distant wind chimes.'],
      ['Sunset Ocean Waves', 'Close-up cinematic shot of waves crashing against volcanic black rocks at golden hour sunset, sea spray glistening in warm amber light, realistic ocean surf and seagull ambient audio.'],
      ['Push-In From Frame 1', 'The camera slowly pushes forward from the first frame, gentle wind rustling the leaves and warm ambient soundscape.'],
    ],
  },
};

const $ = id => document.getElementById(id);
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (v, d = 0) => (v == null || isNaN(v)) ? '–' : Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const toDate = t => typeof t === 'number' ? new Date(t * 1000) : new Date(t);
const hms = t => { const d = toDate(t); return isNaN(d) ? '–' : d.toLocaleTimeString('en-GB', { hour12: false }); };
const hm = t => { const d = toDate(t); return isNaN(d) ? '' : d.toLocaleTimeString('en-GB', { hour12: false, hour: '2-digit', minute: '2-digit' }); };
const avg = (a, f) => a.length ? a.reduce((s, x) => s + f(x), 0) / a.length : null;
const pcieGen = s => { const v = parseFloat(s); return v >= 32 ? 'Gen5' : v >= 16 ? 'Gen4' : v >= 8 ? 'Gen3' : v >= 5 ? 'Gen2' : v ? 'Gen1' : ''; };
const kvRows = rows => rows.map(([k, v]) => `<div class="kv"><span>${esc(k)}</span><b>${esc(v ?? '–')}</b></div>`).join('');

const S = {
  lang: localStorage.getItem('g2e-studio-lang') || 'zh',
  view: 'monitor',
  paused: false,
  interval: 1000,
  timer: null,
  snap: null,
  monTab: 'reqs',
  perf: 'live',
  galFilter: 'all',
  gallery: [],
  chatHistory: [],
  chatImgs: [],
  chatBusy: false,
  q21Imgs: [],
  h3iImgs: [],
  h3First: null,
  h3Last: null,
  h3PollTimer: null,
  lastItems: { qwen21: null, h3_image: null, h3: null },
};

function t(key, ...args) {
  const dict = I18N[S.lang] || I18N.zh;
  const val = dict[key] ?? I18N.zh[key] ?? key;
  return typeof val === 'function' ? val(...args) : val;
}
function isEn() { return S.lang === 'en'; }

function alpha(c, a) {
  if (!c || c[0] !== '#') return c;
  let h = c.slice(1); if (h.length === 3) h = h.split('').map(x => x + x).join('');
  const n = parseInt(h, 16); return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
}
function niceMax(v) {
  if (!(v > 0)) return 1;
  v *= 1.12; const p = Math.pow(10, Math.floor(Math.log10(v))), m = v / p;
  return [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find(s => s >= m - 1e-9) * p;
}
function shortNum(v) {
  const a = Math.abs(v);
  if (a >= 10000) return (v / 1000).toFixed(0) + 'k';
  if (a >= 1000) return (v / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
  if (a >= 10 || Number.isInteger(v)) return String(Math.round(v));
  return v.toFixed(1);
}
function tween(el, to, dec = 0) {
  if (to == null || isNaN(to)) { el.textContent = '–'; el._v = null; return; }
  const from = el._v ?? to; el._v = to;
  cancelAnimationFrame(el._raf);
  const t0 = performance.now(), dur = 380;
  const step = now => {
    const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(from + (to - from) * e, dec);
    if (k < 1) el._raf = requestAnimationFrame(step);
  };
  el._raf = requestAnimationFrame(step);
}
let toastTimer = null;
function toast(msg) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('on'), 2800);
}

/* -------- Sparkline & Rings -------- */
function spark(svg, values, color) {
  const w = Math.max(40, svg.clientWidth || 300), h = 44;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  if (!values.length) { svg.innerHTML = ''; return; }
  const vs = values.length === 1 ? [values[0], values[0]] : values;
  const max = Math.max(...vs) * 1.18 || 1;
  const pts = vs.map((v, i) => [i * w / (vs.length - 1), h - 4 - (v / max) * (h - 10)]);
  let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
  for (let i = 1; i < pts.length; i++) {
    const [x0, y0] = pts[i - 1], [x1, y1] = pts[i], mx = (x0 + x1) / 2;
    d += ` C${mx.toFixed(1)},${y0.toFixed(1)} ${mx.toFixed(1)},${y1.toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`;
  }
  const id = svg.id + 'G';
  svg.innerHTML = `<defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${color}" stop-opacity=".18"/><stop offset="1" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>`
    + `<path d="${d} L${w},${h} L0,${h} Z" fill="url(#${id})"/>`
    + `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.75" stroke-linecap="round"/>`;
}
const RING_C = 2 * Math.PI * 42;
function setRing(id, ratio, val, sub) {
  const r = Math.max(0, Math.min(1, ratio || 0));
  $(id).setAttribute('stroke-dasharray', `${(RING_C * r).toFixed(1)} ${RING_C.toFixed(1)}`);
  const k = id.slice(1);
  $(`g${k}Val`).textContent = val;
  $(`g${k}Sub`).textContent = sub;
}
function legend(el, series) {
  el.innerHTML = series.map(s => `<span><i style="background:${cssv(s.color)}"></i>${esc(s.label)}</span>`).join('');
}

/* -------- Canvas Chart -------- */
function smooth(ctx, pts) {
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) {
    const [x0, y0] = pts[i - 1], [x1, y1] = pts[i], mx = (x0 + x1) / 2;
    ctx.bezierCurveTo(mx, y0, mx, y1, x1, y1);
  }
}
function roundTop(ctx, x, y, w, h, r) {
  ctx.beginPath(); if (h <= 0.5) return;
  r = Math.min(r, h, w / 2);
  ctx.moveTo(x, y + h); ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r); ctx.lineTo(x + w, y + h); ctx.closePath();
}
class Chart {
  constructor(box, opts) {
    this.box = box; this.o = opts; this.data = []; this.mx = null;
    this.cv = document.createElement('canvas'); box.appendChild(this.cv);
    this.tip = document.createElement('div'); this.tip.className = 'tip'; box.appendChild(this.tip);
    this.cv.addEventListener('mousemove', e => { const r = this.cv.getBoundingClientRect(); this.mx = e.clientX - r.left; this.my = e.clientY - r.top; this.draw(); });
    this.cv.addEventListener('mouseleave', () => { this.mx = null; this.tip.style.opacity = 0; this.draw(); });
  }
  set(data, patch) { this.data = data || []; if (patch) Object.assign(this.o, patch); this.draw(); }
  draw() {
    const o = this.o, data = this.data, cv = this.cv;
    const W = this.box.clientWidth, H = this.box.clientHeight; if (!W || !H) return;
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const series = o.series.map(s => ({ ...s, axis: s.axis || 'l', col: cssv(s.color) }));
    const hasR = series.some(s => s.axis === 'r');
    const P = { l: 42, r: hasR ? 38 : 10, t: 10, b: 24 }, pw = W - P.l - P.r, ph = H - P.t - P.b;
    const grid = cssv('--grid'), tert = cssv('--tertiary'), fnt = cssv('--font-text');
    const n = data.length;
    if (!n) {
      ctx.fillStyle = cssv('--muted'); ctx.font = `13px ${fnt}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(o.empty || t('waiting_data'), W / 2, H / 2); this.tip.style.opacity = 0; return;
    }
    const vals = ax => series.filter(s => s.axis === ax).flatMap(s => data.map(d => +s.get(d) || 0));
    const lMax = o.lMax ? o.lMax(data) : niceMax(Math.max(0, ...vals('l')));
    const rMax = hasR ? (o.rMax ? o.rMax(data) : niceMax(Math.max(0, ...vals('r')))) : 1;

    ctx.font = `11px ${fnt}`; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = Math.round(P.t + ph * i / 4) + .5;
      ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(P.l, y); ctx.lineTo(W - P.r, y); ctx.stroke();
      ctx.fillStyle = tert; ctx.textBaseline = 'middle';
      ctx.textAlign = 'right'; ctx.fillText(shortNum(lMax * (1 - i / 4)), P.l - 8, y);
      if (hasR) { ctx.textAlign = 'left'; ctx.fillText(shortNum(rMax * (1 - i / 4)), W - P.r + 8, y); }
    }
    const cat = o.mode === 'cat', slot = pw / n;
    const X = i => cat ? P.l + slot * (i + .5) : (n === 1 ? P.l + pw / 2 : P.l + pw * i / (n - 1));
    const Y = (v, ax) => { const m = ax === 'r' ? rMax : lMax; return P.t + ph * (1 - Math.max(0, Math.min(v, m)) / m); };
    let hi = -1;
    if (this.mx != null && this.mx >= P.l - 10 && this.mx <= W - P.r + 10) {
      hi = cat ? Math.floor((this.mx - P.l) / slot) : Math.round((this.mx - P.l) / pw * (n - 1));
      hi = Math.max(0, Math.min(n - 1, hi));
    }
    ctx.fillStyle = tert; ctx.textBaseline = 'top';
    const ticks = Math.min(n, cat ? 6 : 5);
    for (let k = 0; k < ticks; k++) {
      const i = ticks === 1 ? n - 1 : Math.round(k * (n - 1) / (ticks - 1));
      ctx.textAlign = cat ? 'center' : (k === 0 ? 'left' : k === ticks - 1 ? 'right' : 'center');
      ctx.fillText(o.xLabel(data[i], i), X(i), H - P.b + 8);
    }
    series.filter(s => s.type === 'bar').forEach(s => {
      const bw = Math.max(4, Math.min(20, slot * .56));
      data.forEach((d, i) => {
        const y = Y(+s.get(d) || 0, s.axis), x = X(i) - bw / 2, h = P.t + ph - y;
        ctx.fillStyle = alpha(s.col, hi === i ? 1 : .78); roundTop(ctx, x, y, bw, h, 4); ctx.fill();
      });
    });
    series.filter(s => s.type !== 'bar').forEach(s => {
      const pts = data.map((d, i) => [X(i), Y(+s.get(d) || 0, s.axis)]);
      if (s.type === 'area' && pts.length > 1) {
        const g = ctx.createLinearGradient(0, P.t, 0, P.t + ph);
        g.addColorStop(0, alpha(s.col, .16)); g.addColorStop(1, alpha(s.col, 0));
        ctx.beginPath(); smooth(ctx, pts); ctx.lineTo(pts[pts.length - 1][0], P.t + ph); ctx.lineTo(pts[0][0], P.t + ph); ctx.closePath();
        ctx.fillStyle = g; ctx.fill();
      }
      ctx.beginPath(); smooth(ctx, pts);
      ctx.strokeStyle = s.col; ctx.lineWidth = s.width || 2; ctx.lineJoin = 'round';
      ctx.setLineDash(s.dash || []); ctx.stroke(); ctx.setLineDash([]);
    });
    if (hi < 0) { this.tip.style.opacity = 0; return; }
    const x = X(hi);
    if (!cat) {
      ctx.strokeStyle = cssv('--hairline-2');
      ctx.beginPath(); ctx.moveTo(Math.round(x) + .5, P.t); ctx.lineTo(Math.round(x) + .5, P.t + ph); ctx.stroke();
    }
    const d = data[hi];
    this.tip.innerHTML = `<div class="tip-t">${esc(o.tipTitle ? o.tipTitle(d, hi) : o.xLabel(d, hi))}</div>`
      + series.map(s => `<div class="tip-r"><span><i style="background:${s.col}"></i>${esc(s.label)}</span><b>${esc(s.fmt ? s.fmt(s.get(d)) : fmt(s.get(d), 1))}</b></div>`).join('');
    this.tip.style.opacity = 1;
    const tw = this.tip.offsetWidth, th = this.tip.offsetHeight;
    let tx = x + 14; if (tx + tw > W - 4) tx = x - tw - 14;
    const ty = Math.max(4, Math.min(H - th - 4, (this.my ?? P.t) - th / 2));
    this.tip.style.transform = `translate(${Math.max(4, tx)}px,${ty}px)`;
  }
}

/* ==================== Apply Language ==================== */
function renderPresets() {
  const bind = (boxId, list, inputId) => {
    const el = $(boxId);
    el.innerHTML = list.map(([lbl, p]) => `<button class="chip" data-p="${esc(p)}">${esc(lbl)}</button>`).join('');
    el.querySelectorAll('.chip').forEach(b => b.onclick = () => { $(inputId).value = b.dataset.p; $(inputId).focus(); });
  };
  bind('chatPresets', t('presets_chat'), 'chatInput');
  bind('q21Presets', t('presets_q21'), 'q21Prompt');
  bind('h3iPresets', t('presets_h3i'), 'h3iPrompt');
  bind('h3Presets', t('presets_h3'), 'h3Prompt');
}

function renderH3LoraOptions() {
  const sel = $('h3Lora');
  const cur = sel.value || 'v1.2_8step';
  const mapLabel = id => id === 'v1.2_8step' ? t('lora_v12') : id === 'v1.0_4step' ? t('lora_v10') : t('lora_none');
  const loras = S.snap?.engines?.h3_loras || [
    { id: 'v1.2_8step', steps: 8 },
    { id: 'v1.0_4step', steps: 4 },
    { id: 'none', steps: 50 },
  ];
  sel.innerHTML = loras.map(l => `<option value="${esc(l.id)}" data-steps="${l.steps}">${esc(mapLabel(l.id))}</option>`).join('');
  if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
}

function applyLang(lang) {
  S.lang = (lang === 'en') ? 'en' : 'zh';
  localStorage.setItem('g2e-studio-lang', S.lang);
  document.documentElement.lang = isEn() ? 'en' : 'zh-CN';
  document.querySelectorAll('#segLang button').forEach(b => b.classList.toggle('on', b.dataset.l === S.lang));

  document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll('[data-i18n-ph]').forEach(el => { el.placeholder = t(el.dataset.i18nPh); });
  document.querySelectorAll('[data-i18n-title]').forEach(el => { el.title = t(el.dataset.i18nTitle); });

  renderPresets();
  renderH3LoraOptions();
  renderQ21Thumbs();
  renderH3iThumbs();
  renderH3Frames();
  if (!S.chatHistory.length) {
    $('chatMsgs').innerHTML = `<div class="stage-empty" style="margin:auto"><b>${esc(t('chat_empty_t'))}</b>${esc(t('chat_empty_s'))}</div>`;
  }
  if (S.lastItems.qwen21) $('q21Actions').innerHTML = buildImageActions(S.lastItems.qwen21);
  if (S.lastItems.h3_image) $('h3iActions').innerHTML = buildImageActions(S.lastItems.h3_image);
  if (S.lastItems.h3) $('h3Actions').innerHTML = `<a class="btn sm primary" href="${esc(S.lastItems.h3.url)}" download="${esc(S.lastItems.h3.filename)}">${esc(t('act_dl_mp4'))}</a>`;

  refreshChartLabels();
  renderGallery();
  if (S.snap) {
    renderEngineStrip(S.snap);
    if (S.view === 'monitor') renderMonitor(S.snap);
  }
}
document.querySelectorAll('#segLang button').forEach(b => b.onclick = () => applyLang(b.dataset.l));

/* -------- Navigation & Lightbox -------- */
function switchView(v) {
  S.view = v;
  document.querySelectorAll('#navTabs button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  document.querySelectorAll('.view').forEach(el => el.classList.toggle('on', el.id === 'v-' + v));
  if (v === 'gallery') loadGallery();
  if (v === 'monitor') {
    if (S.snap) renderMonitor(S.snap);
    if (S.monTab === 'logs') fetchLogs();
  }
}
document.querySelectorAll('#navTabs button').forEach(b => b.onclick = () => switchView(b.dataset.v));
$('hwMini').onclick = () => switchView('monitor');
$('btnViewLogs').onclick = () => { switchView('monitor'); switchMonTab('logs'); };

function openLightbox(url, kind, cap) {
  $('lbImg').style.display = kind === 'video' ? 'none' : 'block';
  $('lbVid').style.display = kind === 'video' ? 'block' : 'none';
  if (kind === 'video') { $('lbVid').src = url; $('lbVid').play().catch(() => {}); }
  else { $('lbVid').pause(); $('lbImg').src = url; }
  $('lbCap').textContent = cap || '';
  $('lightbox').classList.add('on');
}
function closeLightbox() { $('lightbox').classList.remove('on'); $('lbVid').pause(); }
$('lightbox').onclick = closeLightbox;
window.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

/* -------- Engine Switching -------- */
async function switchEngine(eid) {
  try {
    const r = await fetch('/api/engine/switch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ engine: eid }),
    });
    const d = await r.json();
    if (!d.ok) return toast(t('req_fail') + ': ' + (d.error || d.message));
    toast(t('t_switching', eid));
    fetchSnap();
  } catch (e) { toast(t('req_fail') + ': ' + e.message); }
}
async function stopEngine() {
  try {
    const r = await fetch('/api/engine/stop', { method: 'POST' });
    const d = await r.json();
    toast(d.ok ? t('t_stopping') : (t('req_fail') + ': ' + (d.error || d.message)));
    fetchSnap();
  } catch (e) { toast(t('req_fail') + ': ' + e.message); }
}
window.switchEngine = switchEngine;
window.stopEngine = stopEngine;
window.closeLightbox = closeLightbox;

function renderEngineStrip(snap) {
  const E = snap.engines || {};
  const items = E.items || [];
  const active = E.active;
  const st = E.state || 'idle';
  const hints = t('eng_hints');

  const dotCls = st === 'loading' || st === 'stopping' ? 'load' : active ? 'ok' : (st === 'error' ? 'err' : '');
  $('stDot').className = 'sdot ' + dotCls;
  $('stText').textContent = st === 'loading' || st === 'stopping'
    ? (isEn() ? t('switching') : (E.status_msg || t('switching')))
    : active ? `${items.find(x => x.id === active)?.name || active} · ${t('running')}` : t('card_idle');

  document.querySelectorAll('#navTabs button').forEach(b => {
    const id = b.dataset.v;
    const it = items.find(x => x.id === id);
    b.classList.toggle('live', !!(it && it.healthy));
    b.classList.toggle('loading', !!(it && it.loading));
  });

  const lb = $('loadBanner');
  if (st === 'loading' || st === 'stopping' || st === 'error') {
    lb.classList.add('on');
    lb.classList.toggle('err', st === 'error');
    $('loadSpin').style.display = st === 'error' ? 'none' : '';
    const tgtName = items.find(x => x.id === E.target)?.name || E.target || '';
    $('loadMsg').textContent = isEn()
      ? (st === 'error' ? `Error: ${E.error || 'failed'}` : st === 'stopping' ? 'Stopping engine & releasing HBM…' : `Loading ${tgtName} to Gaudi2 HBM…`)
      : (E.status_msg || (st === 'error' ? E.error : t('switching')));
    if (E.elapsed_s != null && st !== 'error') {
      $('loadElapsed').style.display = '';
      $('loadSec').textContent = `${fmt(E.elapsed_s, 0)}s`;
    } else {
      $('loadElapsed').style.display = 'none';
    }
    const tail = (snap.log_tail || []).filter(Boolean);
    $('loadLog').textContent = st === 'error' ? (E.error || '') : (tail[tail.length - 1] || '');
  } else {
    lb.classList.remove('on');
  }

  $('engStrip').innerHTML = items.map(it => {
    const cls = it.healthy ? 'active' : it.loading ? 'loading' : '';
    const dot = it.healthy ? 'ok' : it.loading ? 'load' : '';
    const [hbmStr, spdStr] = hints[it.id] || [it.hbm_est, it.speed_hint];
    let btnHtml = '';
    if (it.healthy) {
      btnHtml = `<button class="btn sm danger" onclick="stopEngine()">${esc(t('stop'))}</button>`;
    } else if (it.loading) {
      btnHtml = `<button class="btn sm" disabled>${esc(t('loading'))}</button>`;
    } else {
      btnHtml = `<button class="btn sm primary" ${st === 'loading' || st === 'stopping' ? 'disabled' : ''} onclick="switchEngine('${it.id}')">${esc(t('start'))}</button>`;
    }
    return `<div class="eng-card ${cls}">
      <div class="eng-info" style="cursor:pointer" onclick="switchView('${it.kind === 'chat' ? 'llm' : it.id}')">
        <div class="eng-name"><i class="sdot ${dot}"></i>${esc(it.name)}</div>
        <div class="eng-meta">${esc(hbmStr)} · ${esc(spdStr)}</div>
      </div>
      ${btnHtml}
    </div>`;
  }).join('');

  ['llm', 'qwen21', 'h3_image', 'h3'].forEach(eid => {
    const its = items.filter(x => x.id === eid || (eid === 'llm' && x.kind === 'chat'));
    const el = $('need-' + eid);
    if (el) el.classList.toggle('on', !its.some(it => it.healthy || it.loading));
  });
  const chatOn = items.find(x => x.kind === 'chat' && x.healthy);
  if (chatOn) S.chatModel = chatOn.name;

  if (E.h3_loras && !$('h3Lora').dataset.loaded) {
    $('h3Lora').dataset.loaded = '1';
    renderH3LoraOptions();
  }
}

/* -------- File Reading Helpers -------- */
function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve({ preview: fr.result, data_url: fr.result, name: file.name || 'image.png' });
    fr.onerror = reject;
    fr.readAsDataURL(file);
  });
}
function wireDropzone(dropEl, fileInput, onFiles) {
  dropEl.addEventListener('click', e => { if (!e.target.closest('button')) fileInput.click(); });
  fileInput.addEventListener('change', async () => {
    if (fileInput.files.length) await onFiles(Array.from(fileInput.files));
    fileInput.value = '';
  });
  dropEl.addEventListener('dragover', e => { e.preventDefault(); dropEl.classList.add('over'); });
  dropEl.addEventListener('dragleave', () => dropEl.classList.remove('over'));
  dropEl.addEventListener('drop', async e => {
    e.preventDefault(); dropEl.classList.remove('over');
    const imgs = Array.from(e.dataTransfer.files || []).filter(f => f.type.startsWith('image/'));
    if (imgs.length) await onFiles(imgs);
  });
}

/* -------- Cross-Model Handoff -------- */
function sendToStudio(target, item) {
  const obj = { preview: item.url, gallery_id: item.id, name: item.filename || 'ref.png' };
  if (target === 'qwen21') {
    S.q21Imgs = [obj]; renderQ21Thumbs(); switchView('qwen21'); toast(t('t_sent_q21'));
  } else if (target === 'h3_image') {
    S.h3iImgs = [obj]; renderH3iThumbs(); switchView('h3_image'); toast(t('t_sent_h3i'));
  } else if (target === 'h3_first') {
    S.h3First = obj; renderH3Frames(); switchView('h3'); toast(t('t_sent_h3v'));
  } else if (target === 'llm') {
    S.chatImgs = [obj]; renderChatThumbs(); switchView('llm'); $('chatInput').focus(); toast(t('t_sent_llm'));
  }
}
window.sendToStudio = sendToStudio;

function buildImageActions(item) {
  const j = esc(JSON.stringify({ id: item.id, url: item.url, filename: item.filename }));
  return `
    <button class="btn sm" onclick='sendToStudio("qwen21", ${j})'>${esc(t('act_q21'))}</button>
    <button class="btn sm" onclick='sendToStudio("h3_image", ${j})'>${esc(t('act_h3i'))}</button>
    <button class="btn sm" onclick='sendToStudio("h3_first", ${j})'>${esc(t('act_h3v'))}</button>
    <button class="btn sm" onclick='sendToStudio("llm", ${j})'>${esc(t('act_llm'))}</button>
    <a class="btn sm" href="${esc(item.url)}" download="${esc(item.filename)}">${esc(t('act_dl_png'))}</a>
  `;
}

/* ==================== 1. 27B CHAT + VISION ==================== */
function renderChatThumbs() {
  $('chatThumbs').innerHTML = S.chatImgs.map((im, i) => `
    <div class="thumb"><img src="${esc(im.preview)}"><button class="rm" onclick="S.chatImgs.splice(${i},1);renderChatThumbs()">×</button></div>
  `).join('');
}
$('chatAddImg').onclick = () => $('chatFile').click();
$('chatFile').onchange = async () => {
  for (const f of Array.from($('chatFile').files)) S.chatImgs.push(await readFileAsDataUrl(f));
  $('chatFile').value = '';
  renderChatThumbs();
};
$('chatInput').addEventListener('paste', async e => {
  for (const it of Array.from(e.clipboardData?.items || [])) {
    if (it.type.startsWith('image/')) {
      const f = it.getAsFile();
      if (f) { S.chatImgs.push(await readFileAsDataUrl(f)); renderChatThumbs(); }
    }
  }
});
$('chatClear').onclick = () => {
  S.chatHistory = []; S.chatImgs = []; renderChatThumbs();
  $('chatMsgs').innerHTML = `<div class="stage-empty" style="margin:auto"><b>${esc(t('chat_new_t'))}</b>${esc(t('chat_new_s'))}</div>`;
};

async function sendChat() {
  if (S.chatBusy) return;
  const text = $('chatInput').value.trim();
  if (!text && !S.chatImgs.length) return;
  const imgs = S.chatImgs.slice();
  S.chatImgs = []; renderChatThumbs();
  $('chatInput').value = '';

  const box = $('chatMsgs');
  if (box.querySelector('.stage-empty')) box.innerHTML = '';

  let userContent;
  if (imgs.length) {
    userContent = [];
    for (const im of imgs) {
      if (im.gallery_id) userContent.push({ type: 'gallery_image', gallery_id: im.gallery_id });
      else userContent.push({ type: 'image_url', image_url: { url: im.data_url } });
    }
    userContent.push({ type: 'text', text: text || t('img_only_prompt') });
  } else {
    userContent = text;
  }
  S.chatHistory.push({ role: 'user', content: userContent });

  const uEl = document.createElement('div');
  uEl.className = 'msg user';
  uEl.innerHTML = `<div class="msg-role">${esc(t('role_user'))}</div><div class="msg-bub">${
    imgs.length ? `<div class="msg-imgs">${imgs.map(i => `<img src="${esc(i.preview)}">`).join('')}</div>` : ''
  }${esc(text || t('img_analysis_lbl'))}</div>`;
  box.appendChild(uEl);

  const aEl = document.createElement('div');
  aEl.className = 'msg assistant';
  aEl.innerHTML = `<div class="msg-role">${esc(S.chatModel || 'Assistant')}</div>
    <details class="think" style="display:none"><summary><span>${esc(t('think_title'))}</span><span class="tlen"></span></summary><pre></pre></details>
    <div class="msg-bub typing"></div>
    <div class="msg-meta">${esc(t('wait_resp'))}</div>`;
  box.appendChild(aEl);
  box.scrollTop = box.scrollHeight;

  const thinkBox = aEl.querySelector('.think');
  const thinkPre = aEl.querySelector('.think pre');
  const thinkLen = aEl.querySelector('.tlen');
  const bub = aEl.querySelector('.msg-bub');
  const metaEl = aEl.querySelector('.msg-meta');

  const msgs = [];
  const sys = $('chatSys').value.trim();
  if (sys) msgs.push({ role: 'system', content: sys });
  msgs.push(...S.chatHistory);

  S.chatBusy = true;
  $('chatSend').disabled = true;
  let fullContent = '';
  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: msgs,
        max_tokens: +$('chatMax').value || 2048,
        temperature: +$('chatTemp').value || 0.7,
        enable_thinking: $('chatThink').checked,
      }),
    });
    if (!res.ok || !res.body) throw new Error('HTTP ' + res.status);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf('\n\n')) >= 0) {
        const blk = buf.slice(0, k); buf = buf.slice(k + 2);
        const ln = blk.split('\n').find(l => l.startsWith('data:'));
        if (!ln) continue;
        const ev = JSON.parse(ln.slice(5));
        if (ev.type === 'delta') {
          if (ev.r) {
            thinkBox.style.display = '';
            thinkPre.textContent += ev.r;
            thinkLen.textContent = `${thinkPre.textContent.length} ${t('chars_unit')}`;
          }
          if (ev.c) {
            fullContent += ev.c;
            bub.textContent = fullContent;
          }
          metaEl.textContent = `${t('streaming_meta')} · ${ev.n} chunks · ${(ev.t / 1000).toFixed(2)}s`;
          box.scrollTop = box.scrollHeight;
        } else if (ev.type === 'done') {
          metaEl.textContent = `TTFT ${fmt(ev.ttft_ms, 0)} ms · ${fmt(ev.decode_tps, 1)} tok/s · ${ev.completion_tokens} tok · ${(ev.total_ms / 1000).toFixed(2)} s`;
          if (!fullContent.trim()) bub.textContent = t('think_only_msg');
        } else if (ev.type === 'error') {
          throw new Error(ev.error);
        }
      }
    }
    S.chatHistory.push({ role: 'assistant', content: fullContent });
  } catch (e) {
    bub.innerHTML = `<span style="color:var(--red)">${esc(t('req_fail'))}: ${esc(e.message)} (${esc(t('ensure_27b'))})</span>`;
    metaEl.textContent = t('failed');
    S.chatHistory.pop();
  } finally {
    bub.classList.remove('typing');
    S.chatBusy = false;
    $('chatSend').disabled = false;
    box.scrollTop = box.scrollHeight;
  }
}
$('chatSend').onclick = sendChat;
$('chatInput').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); } });

/* ==================== 2. QWEN-IMAGE-2.1-TURBO ==================== */
function renderQ21Thumbs() {
  $('q21Thumbs').innerHTML = S.q21Imgs.map((im, i) => `
    <div class="thumb"><img src="${esc(im.preview)}"><span class="tag">${esc(t('pic_tag', i + 1))}</span>
    <button class="rm" onclick="event.stopPropagation();S.q21Imgs.splice(${i},1);renderQ21Thumbs()">×</button></div>
  `).join('');
  $('q21ClearImgs').style.display = S.q21Imgs.length ? '' : 'none';
  $('q21ModeBadge').textContent = S.q21Imgs.length ? t('mode_edit_n', S.q21Imgs.length) : t('mode_t2i');
}
wireDropzone($('q21Drop'), $('q21File'), async files => {
  for (const f of files) {
    if (S.q21Imgs.length >= 4) { toast(t('t_max_q21')); break; }
    S.q21Imgs.push(await readFileAsDataUrl(f));
  }
  renderQ21Thumbs();
});
$('q21ClearImgs').onclick = () => { S.q21Imgs = []; renderQ21Thumbs(); };
document.querySelectorAll('#q21Sizes .chip').forEach(b => b.onclick = () => {
  document.querySelectorAll('#q21Sizes .chip').forEach(x => x.classList.toggle('on', x === b));
  $('q21W').value = b.dataset.w; $('q21H').value = b.dataset.h;
});
$('q21RandSeed').onclick = () => { $('q21Seed').value = Math.floor(Math.random() * 1000000); };

$('q21Run').onclick = async () => {
  const prompt = $('q21Prompt').value.trim();
  if (!prompt) return toast(t('t_enter_prompt'));
  $('q21Run').disabled = true;
  $('q21Busy').classList.add('on');
  const t0 = performance.now();
  const tm = setInterval(() => { $('q21BusyTxt').textContent = t('t_q21_busy', ((performance.now() - t0) / 1000).toFixed(1)); }, 100);
  try {
    const r = await fetch('/api/image/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        engine: 'qwen21', prompt,
        width: +$('q21W').value || 1024, height: +$('q21H').value || 1024,
        steps: +$('q21Steps').value || 8, seed: +$('q21Seed').value || 42,
        images: S.q21Imgs.map(im => im.gallery_id ? { gallery_id: im.gallery_id } : { data_url: im.data_url, name: im.name }),
      }),
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || 'failed');
    const item = d.item, tm_ = d.timing || {};
    S.lastItems.qwen21 = item;
    $('q21Empty').style.display = 'none';
    $('q21OutImg').src = item.url;
    $('q21OutImg').style.display = 'block';
    $('q21OutImg').onclick = () => openLightbox(item.url, 'image', `${item.width}×${item.height} · ${item.prompt}`);
    $('q21Timing').innerHTML = `
      <span class="pill">${esc(t('total_t'))} <b>${fmt(d.wall_s, 2)}s</b></span>
      <span class="pill">TE <b>${fmt(tm_.text_encode_s, 2)}s</b></span>
      <span class="pill">DiT <b>${fmt(tm_.denoise_s, 2)}s</b></span>
      <span class="pill">VAE <b>${fmt(tm_.vae_decode_s, 2)}s</b></span>
      <span class="pill">${item.width}×${item.height}</span>`;
    $('q21Actions').innerHTML = buildImageActions(item);
    $('q21Bar').style.display = 'flex';
    toast(t('t_gen_done', fmt(d.wall_s, 2)));
    fetchSnap();
  } catch (e) { toast(t('t_gen_fail', e.message)); }
  finally { clearInterval(tm); $('q21Busy').classList.remove('on'); $('q21Run').disabled = false; }
};

/* ==================== 3. MINIMAX-H3-IMAGE ==================== */
function renderH3iThumbs() {
  $('h3iThumbs').innerHTML = S.h3iImgs.map((im, i) => `
    <div class="thumb"><img src="${esc(im.preview)}"><span class="tag">Picture ${i + 1}</span>
    <button class="rm" onclick="event.stopPropagation();S.h3iImgs.splice(${i},1);renderH3iThumbs()">×</button></div>
  `).join('');
  $('h3iClearImgs').style.display = S.h3iImgs.length ? '' : 'none';
  $('h3iModeBadge').textContent = S.h3iImgs.length ? t('mode_h3i_edit_n', S.h3iImgs.length) : t('mode_t2i');
}
wireDropzone($('h3iDrop'), $('h3iFile'), async files => {
  for (const f of files) {
    if (S.h3iImgs.length >= 16) { toast(t('t_max_h3i')); break; }
    S.h3iImgs.push(await readFileAsDataUrl(f));
  }
  renderH3iThumbs();
});
$('h3iClearImgs').onclick = () => { S.h3iImgs = []; renderH3iThumbs(); };
document.querySelectorAll('#h3iSizes .chip').forEach(b => b.onclick = () => {
  document.querySelectorAll('#h3iSizes .chip').forEach(x => x.classList.toggle('on', x === b));
  $('h3iW').value = b.dataset.w; $('h3iH').value = b.dataset.h;
});
$('h3iRandSeed').onclick = () => { $('h3iSeed').value = Math.floor(Math.random() * 1000000); };

$('h3iRun').onclick = async () => {
  const prompt = $('h3iPrompt').value.trim();
  if (!prompt) return toast(t('t_enter_prompt'));
  $('h3iRun').disabled = true;
  $('h3iBusy').classList.add('on');
  const t0 = performance.now();
  const tm = setInterval(() => { $('h3iBusyTxt').textContent = t('t_h3i_busy', ((performance.now() - t0) / 1000).toFixed(1)); }, 100);
  try {
    const r = await fetch('/api/image/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        engine: 'h3_image', prompt,
        width: +$('h3iW').value || 2048, height: +$('h3iH').value || 2048,
        seed: +$('h3iSeed').value || 101,
        images: S.h3iImgs.map(im => im.gallery_id ? { gallery_id: im.gallery_id } : { data_url: im.data_url, name: im.name }),
      }),
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || 'failed');
    const item = d.item, tm_ = d.timing || {};
    S.lastItems.h3_image = item;
    $('h3iEmpty').style.display = 'none';
    $('h3iOutImg').src = item.url;
    $('h3iOutImg').style.display = 'block';
    $('h3iOutImg').onclick = () => openLightbox(item.url, 'image', `${item.width}×${item.height} · ${item.prompt}`);
    $('h3iTiming').innerHTML = `
      <span class="pill">${esc(t('total_t'))} <b>${fmt(d.wall_s, 2)}s</b></span>
      <span class="pill">${esc(t('encode_t'))} <b>${fmt(tm_.encode_s, 2)}s</b></span>
      <span class="pill">DiT 6× <b>${fmt(tm_.denoise_s, 2)}s</b></span>
      <span class="pill">2D VAE <b>${fmt(tm_.vae_decode_s, 2)}s</b></span>
      <span class="pill">${item.width}×${item.height}</span>`;
    $('h3iActions').innerHTML = buildImageActions(item);
    $('h3iBar').style.display = 'flex';
    toast(t('t_hd_done', fmt(d.wall_s, 2)));
    fetchSnap();
  } catch (e) { toast(t('t_gen_fail', e.message)); }
  finally { clearInterval(tm); $('h3iBusy').classList.remove('on'); $('h3iRun').disabled = false; }
};

/* ==================== 4. MINIMAX-H3 VIDEO ==================== */
function renderH3Frames() {
  $('h3FirstPreview').innerHTML = S.h3First
    ? `<img src="${esc(S.h3First.preview)}" style="width:100%;height:88px;object-fit:cover;border-radius:8px;display:block">`
    : `<small>${esc(t('h3_first_ph'))}</small>`;
  $('h3LastPreview').innerHTML = S.h3Last
    ? `<img src="${esc(S.h3Last.preview)}" style="width:100%;height:88px;object-fit:cover;border-radius:8px;display:block">`
    : `<small>${esc(t('h3_last_ph'))}</small>`;
  $('h3ClrFirst').style.display = S.h3First ? '' : 'none';
  $('h3ClrLast').style.display = S.h3Last ? '' : 'none';
  $('h3ModeBadge').textContent = (S.h3First && S.h3Last) ? t('mode_fl2va') : (S.h3First || S.h3Last) ? t('mode_i2va') : t('mode_t2va');
}
wireDropzone($('h3DropFirst'), $('h3FileFirst'), async fs => { S.h3First = await readFileAsDataUrl(fs[0]); renderH3Frames(); });
wireDropzone($('h3DropLast'), $('h3FileLast'), async fs => { S.h3Last = await readFileAsDataUrl(fs[0]); renderH3Frames(); });
$('h3ClrFirst').onclick = () => { S.h3First = null; renderH3Frames(); };
$('h3ClrLast').onclick = () => { S.h3Last = null; renderH3Frames(); };
document.querySelectorAll('#h3Sizes .chip').forEach(b => b.onclick = () => {
  document.querySelectorAll('#h3Sizes .chip').forEach(x => x.classList.toggle('on', x === b));
  $('h3W').value = b.dataset.w; $('h3H').value = b.dataset.h;
});
$('h3Dur').oninput = () => { $('h3DurVal').textContent = Number($('h3Dur').value).toFixed(1) + ' s'; };
$('h3Lora').onchange = () => {
  const opt = $('h3Lora').selectedOptions[0];
  if (opt && opt.dataset.steps) $('h3Steps').value = opt.dataset.steps;
  else if ($('h3Lora').value === 'v1.0_4step') $('h3Steps').value = 4;
  else if ($('h3Lora').value === 'v1.2_8step') $('h3Steps').value = 8;
  else $('h3Steps').value = 50;
};
$('h3RandSeed').onclick = () => { $('h3Seed').value = Math.floor(Math.random() * 1000000); };

$('h3Run').onclick = async () => {
  const prompt = $('h3Prompt').value.trim();
  if (!prompt) return toast(t('t_enter_vprompt'));
  $('h3Run').disabled = true;
  $('h3Busy').classList.add('on');
  $('h3BusyTxt').textContent = t('t_submitting_vid');
  $('h3BusySub').textContent = '';
  const t0 = performance.now();
  try {
    const r = await fetch('/api/video/submit', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        width: +$('h3W').value || 1280, height: +$('h3H').value || 768,
        duration: +$('h3Dur').value || 5.0, fps: 24,
        steps: +$('h3Steps').value || 8, lora: $('h3Lora').value, seed: +$('h3Seed').value || 42,
        first_frame: S.h3First ? (S.h3First.gallery_id ? { gallery_id: S.h3First.gallery_id } : { data_url: S.h3First.data_url, name: S.h3First.name }) : null,
        last_frame: S.h3Last ? (S.h3Last.gallery_id ? { gallery_id: S.h3Last.gallery_id } : { data_url: S.h3Last.data_url, name: S.h3Last.name }) : null,
      }),
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || 'failed');
    const jid = d.id;
    clearInterval(S.h3PollTimer);
    S.h3PollTimer = setInterval(async () => {
      const elap = ((performance.now() - t0) / 1000).toFixed(1);
      try {
        const sr = await fetch('/api/video/status/' + encodeURIComponent(jid));
        const st = await sr.json();
        if (st.status === 'completed' && st.item) {
          clearInterval(S.h3PollTimer);
          const item = st.item, tm_ = st.timing || {};
          S.lastItems.h3 = item;
          $('h3Busy').classList.remove('on');
          $('h3Run').disabled = false;
          $('h3Empty').style.display = 'none';
          $('h3OutVid').src = item.url;
          $('h3OutVid').style.display = 'block';
          $('h3OutVid').play().catch(() => {});
          $('h3Timing').innerHTML = `
            <span class="pill">${esc(t('total_t'))} <b>${fmt(tm_.total_s, 1)}s</b></span>
            <span class="pill">DiT <b>${fmt(tm_.denoise_s, 1)}s</b></span>
            <span class="pill">VideoVAE <b>${fmt(tm_.vvae_s, 1)}s</b></span>
            <span class="pill">AudioVAE <b>${fmt(tm_.avae_s, 2)}s</b></span>
            <span class="pill">${item.width}×${item.height} · ${item.num_frames || ''}${esc(t('frames_u'))}</span>`;
          $('h3Actions').innerHTML = `<a class="btn sm primary" href="${esc(item.url)}" download="${esc(item.filename)}">${esc(t('act_dl_mp4'))}</a>`;
          $('h3Bar').style.display = 'flex';
          toast(t('t_vid_done', fmt(tm_.total_s, 1)));
          fetchSnap();
        } else if (st.status === 'failed') {
          clearInterval(S.h3PollTimer);
          $('h3Busy').classList.remove('on');
          $('h3Run').disabled = false;
          toast(t('t_vid_fail', st.error || 'unknown'));
        } else {
          $('h3BusyTxt').textContent = t('t_h3_busy', st.status, elap);
          $('h3BusySub').textContent = t('t_h3_busy_sub', jid);
        }
      } catch (e) {}
    }, 1500);
  } catch (e) {
    $('h3Busy').classList.remove('on');
    $('h3Run').disabled = false;
    toast(t('t_vid_fail', e.message));
  }
};

/* ==================== 5. GALLERY ==================== */
async function loadGallery() {
  try {
    const r = await fetch('/api/gallery?limit=100', { cache: 'no-store' });
    const d = await r.json();
    S.gallery = d.items || [];
    $('cntGal').textContent = S.gallery.length;
    renderGallery();
  } catch (e) {}
}
function renderGallery() {
  const list = S.galFilter === 'all' ? S.gallery : S.gallery.filter(x => x.engine === S.galFilter);
  if (!list.length) {
    $('galGrid').innerHTML = `<div class="empty" style="grid-column:1/-1"><b>${esc(t('gal_empty_t'))}</b>${esc(t('gal_empty_s'))}</div>`;
    return;
  }
  const tagMap = { qwen21: 'Qwen 2.1', h3_image: t('nav_h3i'), h3: t('nav_h3') };
  $('galGrid').innerHTML = list.map(it => {
    const isVid = it.kind === 'video';
    const refObj = esc(JSON.stringify({ id: it.id, url: it.url, filename: it.filename }));
    const durStr = it.wall_s ? `${fmt(it.wall_s, 1)}s` : (it.timing?.total_s ? `${fmt(it.timing.total_s, 1)}s` : '');
    return `<div class="gal-card">
      <div class="gal-media" onclick="openLightbox('${esc(it.url)}','${isVid ? 'video' : 'image'}',${esc(JSON.stringify(it.prompt || ''))})">
        ${isVid ? `<video src="${esc(it.url)}" muted loop onmouseover="this.play()" onmouseout="this.pause()"></video>` : `<img src="${esc(it.url)}" loading="lazy" alt="">`}
        <span class="gal-badge">${esc(tagMap[it.engine] || it.engine)} · ${esc(it.mode || '')}</span>
      </div>
      <div class="gal-body">
        <div class="gal-prompt" title="${esc(it.prompt || '')}">${esc(it.prompt || t('no_prompt'))}</div>
        <div class="gal-meta"><span>${it.width || ''}×${it.height || ''} ${durStr ? '· ' + durStr : ''}</span><span>${hms(it.created_at)}</span></div>
        <div class="gal-acts">
          ${!isVid ? `<button class="btn sm" onclick='sendToStudio("h3_image",${refObj})'>${esc(t('gal_act_h3i'))}</button>` : ''}
          ${!isVid ? `<button class="btn sm" onclick='sendToStudio("qwen21",${refObj})'>${esc(t('gal_act_q21'))}</button>` : ''}
          ${!isVid ? `<button class="btn sm" onclick='sendToStudio("h3_first",${refObj})'>${esc(t('gal_act_vid'))}</button>` : ''}
          ${!isVid ? `<button class="btn sm" onclick='sendToStudio("llm",${refObj})'>${esc(t('gal_act_llm'))}</button>` : ''}
          <a class="btn sm" href="${esc(it.url)}" download="${esc(it.filename)}">${esc(t('download'))}</a>
          <button class="btn sm danger" onclick="delGallery('${esc(it.id)}')">${esc(t('del'))}</button>
        </div>
      </div>
    </div>`;
  }).join('');
}
async function delGallery(id) {
  if (!confirm(t('confirm_del'))) return;
  await fetch('/api/gallery/' + encodeURIComponent(id), { method: 'DELETE' });
  loadGallery();
}
window.delGallery = delGallery;
window.openLightbox = openLightbox;
document.querySelectorAll('#galFilter button').forEach(b => b.onclick = () => {
  document.querySelectorAll('#galFilter button').forEach(x => x.classList.toggle('on', x === b));
  S.galFilter = b.dataset.f; renderGallery();
});
$('galRefresh').onclick = loadGallery;

/* ==================== 6. HARDWARE MONITOR ==================== */
function getHwSeries() {
  return [
    { label: t('chart_power'), color: '--orange', type: 'area', get: d => d.power, fmt: v => fmt(v) + ' W' },
    { label: t('chart_util'), color: '--blue', axis: 'r', get: d => d.util, fmt: v => fmt(v) + ' %' },
    { label: t('chart_temp'), color: '--red', axis: 'r', get: d => d.temp, fmt: v => fmt(v) + ' °C' },
  ];
}
function getKvSeries() {
  return [
    { label: t('chart_hbm'), color: '--purple', type: 'area', get: d => d.hbm_gib || 0, fmt: v => fmt(v, 1) + ' GiB' },
    { label: 'KV Cache', color: '--blue', get: d => d.kv_gib || 0, fmt: v => fmt(v, 2) + ' GiB' },
    { label: 'Host Prefix', color: '--green', get: d => d.prefix_gib || 0, fmt: v => fmt(v, 2) + ' GiB' },
  ];
}
function getPerfSeries(mode) {
  if (mode === 'live') return [
    { label: t('chart_live_dec'), color: '--blue', type: 'area', get: d => d.decode_tps || 0, fmt: v => fmt(v, 1) + ' tok/s' },
    { label: t('chart_live_pre'), color: '--green', axis: 'r', get: d => d.prefill_tps || 0, fmt: v => fmt(v) + ' tok/s' },
  ];
  if (mode === 'decode') return [
    { label: 'Decode', color: '--blue', type: 'bar', get: d => d.decode_tps, fmt: v => fmt(v, 1) + ' tok/s' },
    { label: t('chart_spec'), color: '--indigo', axis: 'r', get: d => d.tok_per_round, fmt: v => fmt(v, 2) + ' tok/r' },
  ];
  if (mode === 'prefill') return [
    { label: 'Prefill', color: '--blue', type: 'bar', get: d => d.prefill_tps, fmt: v => fmt(v) + ' tok/s' },
    { label: 'Prompt', color: '--green', axis: 'r', get: d => d.prompt_tokens, fmt: v => fmt(v) + ' tok' },
  ];
  return [
    { label: 'TTFT', color: '--orange', type: 'bar', get: d => d.prompt_ms, fmt: v => fmt(v) + ' ms' },
    { label: t('chart_hit'), color: '--green', axis: 'r', get: d => d.cached_pct, fmt: v => fmt(v, 1) + ' %' },
  ];
}

const chHw = new Chart($('chHw'), {
  series: getHwSeries(), xLabel: d => hms(d.t),
  lMax: d => niceMax(Math.max(100, ...d.map(x => x.power || 0))), rMax: () => 100,
});
const chKv = new Chart($('chKv'), { series: getKvSeries(), xLabel: d => hms(d.t), lMax: () => 96 });
const chPerf = new Chart($('chPerf'), {
  mode: 'time', series: getPerfSeries('live'), empty: t('waiting_data'),
  xLabel: d => hms(d.t), tipTitle: d => hms(d.t),
  lMax: d => niceMax(Math.max(100, ...d.map(x => x.decode_tps || 0))),
  rMax: d => niceMax(Math.max(1000, ...d.map(x => x.prefill_tps || 0))),
});

function refreshChartLabels() {
  chHw.o.series = getHwSeries();
  chKv.o.series = getKvSeries();
  chPerf.o.series = getPerfSeries(S.perf);
  chPerf.o.empty = t('waiting_data');
  legend($('lgHw'), chHw.o.series);
  legend($('lgKv'), chKv.o.series);
  legend($('lgPerf'), chPerf.o.series);
}
function setPerf(mode) {
  S.perf = mode;
  document.querySelectorAll('#segPerf button').forEach(b => b.classList.toggle('on', b.dataset.v === mode));
  chPerf.o.series = getPerfSeries(mode);
  chPerf.o.mode = mode === 'live' ? 'time' : 'cat';
  chPerf.o.xLabel = mode === 'live' ? (d => hms(d.t)) : (d => hm(d.ts));
  chPerf.o.lMax = mode === 'live' ? (d => niceMax(Math.max(100, ...d.map(x => x.decode_tps || 0)))) : null;
  chPerf.o.rMax = mode === 'live' ? (d => niceMax(Math.max(1000, ...d.map(x => x.prefill_tps || 0)))) : null;
  legend($('lgPerf'), chPerf.o.series);
  if (S.snap) chPerf.set(mode === 'live' ? (S.snap.history || []) : (S.snap.requests || []).slice(0, 40).reverse());
}
document.querySelectorAll('#segPerf button').forEach(b => b.onclick = () => setPerf(b.dataset.v));

let slotKey = '';
function renderMonitor(snap) {
  const E = snap.engine || {}, st = E.stats || {}, sl = st.slots || {}, pc = st.prefix_cache || {};
  const hw = snap.hardware || {}, host = snap.host || {}, bn = snap.banner || {};
  const reqs = snap.requests || [], hist = snap.history || [];
  const mode = E.mode || 'idle';
  const en = isEn();

  $('chModel').textContent = E.model || t('not_loaded');
  $('chEngine').textContent = mode === 'idle' ? `Gaudi2 ${t('idle')}` : `${E.url?.replace(/^https?:\/\//, '')} · Direct MME + SRAM`;
  $('hwSub').textContent = `Intel Gaudi2 ${hw.name || 'HL-225'}`;
  const chatMode = mode === 'llm' || mode === 'inkling';
  $('chCtx').textContent = chatMode ? (E.max_model_len ? Math.round(E.max_model_len / 1024) + 'K' : '256K')
    : (st.completed != null ? (en ? `${st.completed} done` : `已完成 ${st.completed}`) : t('idle'));
  $('lastUpd').textContent = hms(snap.timestamp || Date.now() / 1000);

  if (mode === 'qwen21' || mode === 'h3_image') {
    const jobs = (st.recent_jobs || []).slice().reverse(), j0 = jobs[0];
    $('kl0').textContent = en ? 'Image Latency' : '图像总耗时'; $('ku0').textContent = 's';
    tween($('vDecode'), j0 ? j0.total_s : 0, 2);
    $('bDecode').textContent = j0 ? `${j0.mode} · ${j0.width}×${j0.height}` : (en ? 'Ready' : '等待出图');
    $('mDecAvg').textContent = jobs.length ? fmt(avg(jobs, j => j.total_s), 2) + 's' : '–';
    $('mDecMax').textContent = jobs.length ? fmt(Math.min(...jobs.map(j => j.total_s)), 2) + 's' : '–';
    spark($('spDecode'), jobs.slice().reverse().map(j => j.total_s || 0), cssv('--blue'));

    $('kl1').textContent = en ? 'DiT Denoise' : 'DiT 扩散耗时'; $('ku1').textContent = 's';
    tween($('vPrefill'), j0 ? j0.denoise_s : 0, 2);
    $('bPrefill').textContent = mode === 'qwen21' ? '8-step Turbo' : '6-step PDD';
    $('mPreMax').textContent = j0 ? `${j0.width}×${j0.height}` : '–';
    $('mPreLen').textContent = j0 ? (j0.mode === 'Edit' ? `${j0.n_ref || j0.n_cond || 1} ref` : 'T2I') : '–';
    spark($('spPrefill'), jobs.slice().reverse().map(j => j.denoise_s || 0), cssv('--cyan'));

    $('kl2').textContent = en ? 'VAE Decode' : 'VAE 解码耗时'; $('ku2').textContent = 's';
    tween($('vTtft'), j0 ? j0.vae_decode_s : 0, 2);
    $('mTtftAvg').textContent = j0 ? `${t('encode_t')} ${fmt(j0.encode_s ?? j0.text_encode_s ?? 0, 2)}s` : '–';
    $('mTtftHit').textContent = mode === 'h3_image' ? '24L ViT 2D' : 'RGBA KL';
    spark($('spTtft'), jobs.slice().reverse().map(j => j.vae_decode_s || 0), cssv('--orange'));

    $('kl3').textContent = en ? 'Completed' : '累计出图'; $('ku3').textContent = en ? 'imgs' : '张';
    tween($('vSpec'), st.completed || 0, 0);
    $('bSpec').textContent = mode === 'qwen21' ? 'Qwen 2.1' : 'H3-Image';
    $('mSpecAvg').textContent = `HBM ${fmt((hw.mem_used_mib || 0) / 1024, 1)} GiB`;
    $('mSpecRounds').textContent = `AIP ${hw.util_pct ?? 0}%`;
    spark($('spSpec'), hist.slice(-60).map(p => p.util || 0), cssv('--indigo'));

    $('cntReq').textContent = jobs.length;
    $('reqHead').innerHTML = en
      ? '<th>Time</th><th>Mode</th><th>Size</th><th>Encode</th><th>DiT</th><th>VAE</th><th>Total</th>'
      : '<th>时间</th><th>模式</th><th>分辨率</th><th>编码</th><th>DiT 扩散</th><th>VAE 解码</th><th>总耗时</th>';
    $('tbReq').innerHTML = jobs.length ? jobs.map(j => `<tr>
      <td>${hms(j.ts)}</td><td><span class="badge ok">${esc(j.mode || 'T2I')}</span></td>
      <td>${j.width}×${j.height}</td><td>${fmt(j.encode_s ?? j.text_encode_s ?? 0, 2)}s</td>
      <td>${fmt(j.denoise_s, 2)}s</td><td>${fmt(j.vae_decode_s, 2)}s</td><td><b>${fmt(j.total_s, 2)}s</b></td>
    </tr>`).join('') : `<tr><td colspan="7"><div class="empty"><b>${en ? 'No images generated yet' : '暂无出图记录'}</b></div></td></tr>`;
  } else if (mode === 'h3') {
    const hjobs = (st.recent_jobs || []).slice().reverse(), hj0 = hjobs[0], aj = st.active_job;
    $('kl0').textContent = en ? 'Video Total' : '视频总耗时'; $('ku0').textContent = 's';
    tween($('vDecode'), aj ? aj.elapsed_s : (hj0 ? hj0.total_s : st.last_total_s || 0), 1);
    $('bDecode').textContent = aj ? `${en ? 'Running' : '生成中'} ${aj.width}×${aj.height}` : `${st.completed || 0} · Q ${st.queued || 0}`;
    $('mDecAvg').textContent = hjobs.length ? fmt(avg(hjobs, j => j.total_s), 1) + 's' : '–';
    $('mDecMax').textContent = hjobs.length ? fmt(Math.min(...hjobs.map(j => j.total_s)), 1) + 's' : '–';
    spark($('spDecode'), hjobs.slice().reverse().map(j => j.total_s || 0), cssv('--blue'));

    $('kl1').textContent = en ? 'DiT Denoise' : 'DiT 扩散耗时'; $('ku1').textContent = 's';
    tween($('vPrefill'), hj0 ? hj0.denoise_s : 0, 1);
    $('bPrefill').textContent = hj0 ? `${hj0.steps || 8} steps` : t('idle');
    $('mPreMax').textContent = hj0 ? `${hj0.width}×${hj0.height}` : '–';
    $('mPreLen').textContent = hj0 ? `${hj0.num_frames}${t('frames_u')}` : '–';
    spark($('spPrefill'), hjobs.slice().reverse().map(j => j.denoise_s || 0), cssv('--cyan'));

    $('kl2').textContent = en ? 'VideoVAE Decode' : 'VideoVAE 解码'; $('ku2').textContent = 's';
    tween($('vTtft'), hj0 ? hj0.vvae_s : 0, 1);
    $('mTtftAvg').textContent = hj0 ? `Audio ${fmt(hj0.avae_s, 2)}s` : '–';
    $('mTtftHit').textContent = hj0 ? `Mux ${fmt(hj0.mux_s, 1)}s` : '–';
    spark($('spTtft'), hjobs.slice().reverse().map(j => j.vvae_s || 0), cssv('--orange'));

    $('kl3').textContent = en ? 'Videos Done' : '累计视频'; $('ku3').textContent = en ? 'clips' : '条';
    tween($('vSpec'), st.completed || 0, 0);
    $('bSpec').textContent = 'MiniMax-H3';
    $('mSpecAvg').textContent = `${en ? 'Run' : '运行'} ${st.running || 0}`;
    $('mSpecRounds').textContent = `${en ? 'Q' : '排队'} ${st.queued || 0}`;
    spark($('spSpec'), hist.slice(-60).map(p => p.util || 0), cssv('--indigo'));

    $('cntReq').textContent = hjobs.length;
    $('reqHead').innerHTML = en
      ? '<th>Time</th><th>Size</th><th>Frames</th><th>Text</th><th>DiT</th><th>VideoVAE</th><th>AudioVAE</th><th>Mux</th><th>Total</th>'
      : '<th>时间</th><th>分辨率</th><th>帧数</th><th>文本编码</th><th>DiT 扩散</th><th>VideoVAE</th><th>AudioVAE</th><th>封装</th><th>总耗时</th>';
    $('tbReq').innerHTML = hjobs.length ? hjobs.map(j => `<tr>
      <td>${hms(j.ts)}</td><td>${j.width}×${j.height}</td><td>${j.num_frames}${t('frames_u')}</td>
      <td>${fmt(j.encode_s, 2)}s</td><td>${fmt(j.denoise_s, 1)}s</td><td>${fmt(j.vvae_s, 1)}s</td>
      <td>${fmt(j.avae_s, 2)}s</td><td>${fmt(j.mux_s, 1)}s</td><td><b>${fmt(j.total_s, 1)}s</b></td>
    </tr>`).join('') : `<tr><td colspan="9"><div class="empty"><b>${en ? 'No videos generated yet' : '暂无视频记录'}</b></div></td></tr>`;
  } else {
    const lv = snap.live || {}, chron = reqs.slice(0, 40).reverse(), last = reqs[0], recent = reqs.slice(0, 20);
    const liveDec = lv.supported && ((lv.decode_tps || 0) > 0 || (lv.decoding_slots || 0) > 0);
    const livePre = lv.supported && ((lv.prefill_tps || 0) > 0 || (lv.prefilling_slots || 0) > 0);
    $('kl0').textContent = en ? 'Decode Speed' : 'Decode 吞吐'; $('ku0').textContent = 'tok/s';
    tween($('vDecode'), lv.supported ? (lv.decode_tps || 0) : last?.decode_tps, 1);
    $('bDecode').textContent = liveDec ? `${en ? 'Live' : '实时'} · ${Math.max(1, lv.decoding_slots)}` : `${en ? 'Last' : '上次'} ${last ? fmt(last.decode_tps, 1) : '–'}`;
    $('mDecAvg').textContent = fmt(avg(recent, r => r.decode_tps), 1);
    $('mDecMax').textContent = reqs.length ? fmt(Math.max(...reqs.map(r => r.decode_tps)), 1) : '–';
    spark($('spDecode'), hist.slice(-60).map(p => p.decode_tps || 0), cssv('--blue'));

    const pre = reqs.filter(r => r.prefill_tps > 0);
    $('kl1').textContent = en ? 'Prefill Speed' : 'Prefill 吞吐'; $('ku1').textContent = 'tok/s';
    tween($('vPrefill'), livePre ? lv.prefill_tps : (pre[0]?.prefill_tps ?? 0), 0);
    $('bPrefill').textContent = livePre ? (en ? 'Live' : '实时') : (en ? 'Latest' : '最近一次');
    $('mPreMax').textContent = pre.length ? fmt(Math.max(...pre.map(r => r.prefill_tps))) : '–';
    $('mPreLen').textContent = pre[0] ? fmt(pre[0].prompt_tokens) : '–';
    spark($('spPrefill'), hist.slice(-60).map(p => p.prefill_tps || 0), cssv('--cyan'));

    $('kl2').textContent = en ? 'First Token TTFT' : '首字延迟 TTFT'; $('ku2').textContent = 'ms';
    tween($('vTtft'), last?.prompt_ms, 0);
    $('mTtftAvg').textContent = recent.length ? fmt(avg(recent, r => r.prompt_ms)) + ' ms' : '–';
    $('mTtftHit').textContent = fmt((st.token_hit_rate || 0) * 100, 1) + '%';
    spark($('spTtft'), chron.map(r => r.prompt_ms), cssv('--orange'));

    $('kl3').textContent = en ? 'Speculative Accept' : '投机接受效率'; $('ku3').textContent = en ? 'tok/rnd' : 'tok/轮';
    tween($('vSpec'), (liveDec && lv.tok_per_round > 0) ? lv.tok_per_round : last?.tok_per_round, 2);
    $('bSpec').textContent = 'DSpine-b16';
    $('mSpecAvg').textContent = fmt(avg(recent, r => r.tok_per_round), 2);
    $('mSpecRounds').textContent = last ? fmt(last.rounds) : '–';
    spark($('spSpec'), chron.map(r => r.tok_per_round), cssv('--indigo'));

    $('cntReq').textContent = reqs.length;
    $('reqHead').innerHTML = en
      ? '<th>Time</th><th>Prompt</th><th>Prefix Hit</th><th>TTFT</th><th>Prefill</th><th>Output</th><th>Decode</th><th>Spec</th><th>Finish</th>'
      : '<th>时间</th><th>Prompt</th><th>前缀命中</th><th>TTFT</th><th>Prefill</th><th>输出</th><th>Decode</th><th>投机</th><th>状态</th>';
    const maxDec = Math.max(...reqs.map(r => r.decode_tps), 1);
    $('tbReq').innerHTML = reqs.length ? reqs.map(r => `<tr>
      <td>${hms(r.ts)}</td><td>${fmt(r.prompt_tokens)} tok</td>
      <td>${r.cached_tokens > 0 ? `<span class="badge ok">${fmt(r.cached_pct)}%</span>` : `<span class="badge dim">${en ? 'Miss' : '未命中'}</span>`}</td>
      <td>${fmt(r.prompt_ms)} ms</td><td>${r.prefill_tps > 0 ? fmt(r.prefill_tps) + ' tok/s' : '–'}</td>
      <td>${fmt(r.gen_tokens)} tok</td>
      <td><span class="bar-cell"><i style="--w:${(r.decode_tps / maxDec * 100).toFixed(1)}%"></i><b>${fmt(r.decode_tps, 1)}</b></span></td>
      <td>${fmt(r.tok_per_round, 2)}</td><td><span class="badge ok">${esc(r.finish)}</span></td>
    </tr>`).join('') : `<tr><td colspan="9"><div class="empty"><b>${en ? 'No LLM requests yet' : '暂无 LLM 请求记录'}</b></div></td></tr>`;
  }

  const pmax = hw.power_max_w || 600, cmax = hw.clock_max_mhz || hw.clock_mhz || 1;
  const powSub = (hw.power_54v_w != null && hw.power_12v_w != null)
    ? `54V ${fmt(hw.power_54v_w)}W + 12V ${fmt(hw.power_12v_w)}W` : `${t('limit')} ${fmt(pmax)} W`;
  setRing('rUtil', (hw.util_pct || 0) / 100, hw.util_pct == null ? '–' : fmt(hw.util_pct), hw.util_pct ? t('computing') : t('idle'));
  setRing('rPow', (hw.power_w || 0) / pmax, hw.power_w == null ? '–' : fmt(hw.power_w), powSub);
  setRing('rTemp', (hw.temp_c || 0) / 100, hw.temp_c == null ? '–' : fmt(hw.temp_c), hw.temp_c == null ? '–' : hw.temp_c < 65 ? t('normal') : hw.temp_c < 85 ? t('warm') : t('hot'));
  setRing('rClk', (hw.clock_mhz || 0) / cmax, hw.clock_mhz == null ? '–' : fmt(hw.clock_mhz), `${t('peak')} ${fmt(cmax)}`);
  const hbmUsed = (hw.mem_used_mib || 0) / 1024, hbmTot = (hw.mem_total_mib || 98304) / 1024;
  $('hwName').textContent = `${hw.name || 'HL-225'} · ${hw.available ? (en ? 'hl-smi live telemetry' : 'hl-smi 实时遥测') : t('unavail')}`;
  $('sHbm').textContent = `${fmt(hbmUsed, 1)} / ${fmt(hbmTot, 0)} GiB`;
  $('sPcie').textContent = hw.pcie_speed ? `${pcieGen(hw.pcie_speed)} ×${hw.pcie_width}` : '–';
  $('sDrv').textContent = hw.driver_ver || '–';
  $('sReset').textContent = hw.hard_reset_cnt ?? '–';
  chHw.set(hist);

  const tot = sl.total || 8, act = sl.active || 0, q = sl.queued || st.queued || 0;
  const key = `${tot}/${act}`;
  if (key !== slotKey) {
    slotKey = key;
    $('slots').innerHTML = Array.from({ length: tot }, (_, i) => `<div class="slot ${i < act ? 'on' : ''}"><span>Slot ${i}</span><i class="sd"></i></div>`).join('');
  }
  $('capSub').textContent = chatMode ? (en ? `${act} / ${tot} active slots` : `${act} / ${tot} 活跃槽位`)
    : (E.healthy ? (en ? `${E.model} resident on card` : `${E.model} 单卡独占常驻`) : `Gaudi2 ${t('idle')}`);
  $('capQueue').textContent = q > 0 ? (en ? `Queued ${q}` : `排队 ${q}`) : (E.healthy ? t('online') : t('idle'));
  $('capQueue').className = 'badge ' + (q > 0 ? 'warn' : (E.healthy ? 'ok' : 'dim'));

  $('mtHbm').textContent = `${fmt(hbmUsed, 1)} / ${fmt(hbmTot, 0)} GiB`;
  $('fHbm').style.width = Math.min(100, hbmUsed / hbmTot * 100) + '%';
  const kvU = sl.kv_used_gib || 0, kvB = sl.kv_budget_gib || 0;
  $('mtKv').textContent = `${fmt(kvU, 1)} / ${fmt(kvB, 1)} GiB`;
  $('fKv').style.width = (kvB ? Math.min(100, kvU / kvB * 100) : 0) + '%';
  $('mtPc').textContent = `${fmt(pc.host_gib || 0, 1)} GiB`;
  $('fPc').style.width = Math.min(100, (pc.host_gib || 0) / 64 * 100) + '%';
  $('mtRam').textContent = `${fmt(host.mem_used_gib, 1)} / ${fmt(host.mem_total_gib, 1)} GiB`;
  $('fRam').style.width = (host.mem_total_gib ? Math.min(100, host.mem_used_gib / host.mem_total_gib * 100) : 0) + '%';

  $('sReq').textContent = fmt(st.requests ?? st.completed ?? 0);
  $('sCpu').textContent = `${fmt(host.cpu_pct, 1)} %`;
  $('sShm').textContent = `${fmt(host.shm_used_gib, 1)} GiB`;
  $('sGal').textContent = fmt(snap.gallery_count || 0);

  $('kvNow').textContent = `HBM ${fmt(hbmUsed, 1)} GiB`;
  chKv.set(hist);
  chPerf.set(S.perf === 'live' ? hist : reqs.slice(0, 40).reverse());

  $('iHw').innerHTML = kvRows([
    [en ? 'Device' : '设备型号', `${hw.name || '–'} (${hw.device_type || '–'})`],
    [en ? 'Driver' : '驱动版本', hw.driver_ver],
    [en ? 'SoC Clock' : 'SoC 频率', hw.clock_mhz != null ? `${fmt(hw.clock_mhz)} / ${fmt(hw.clock_max_mhz)} MHz` : '–'],
    [en ? 'HBM Memory' : 'HBM 显存', `${fmt(hbmUsed, 1)} / ${fmt(hbmTot, 0)} GiB`],
    [en ? 'Temp / Power' : '温度 / 功耗', `${hw.temp_c ?? '–'} °C · ${hw.power_w ?? '–'} / ${hw.power_max_w ?? '–'} W`],
    [en ? 'PCIe Link' : 'PCIe 链路', hw.pcie_speed ? `${hw.pcie_speed} ×${hw.pcie_width}` : '–'],
  ]);
  $('iEng').innerHTML = kvRows([
    [en ? 'Active Model' : '当前引擎', E.model || t('not_running')],
    [en ? 'Endpoint' : 'API 地址', E.url],
    [en ? 'Mode' : '模式', mode],
    [en ? 'Backend' : '执行后端', bn.direct_engine || 'Direct MME + 24 TPC + SRAM'],
    [en ? 'KV Budget' : 'KV 预算', bn.kv_summary || '–'],
  ]);
  $('iSys').innerHTML = kvRows([
    [en ? 'Host CPU' : '宿主机 CPU', `${fmt(host.cpu_pct, 1)} % · load ${host.load1 ?? '–'}`],
    [en ? 'Host RAM' : '宿主机内存', `${fmt(host.mem_used_gib, 1)} / ${fmt(host.mem_total_gib, 1)} GiB`],
    [en ? '/dev/shm RAM Disk' : '/dev/shm 内存盘', `${fmt(host.shm_used_gib, 1)} / ${fmt(host.shm_total_gib, 1)} GiB`],
    [en ? 'Gallery Path' : '作品画廊路径', '/mnt/sas/outputs/g2e_studio'],
  ]);
}

function renderTopMiniHw(snap) {
  const hw = snap.hardware || {};
  $('mnUtil').textContent = hw.util_pct != null ? `${fmt(hw.util_pct)}%` : '–';
  $('mnHbm').textContent = hw.mem_used_mib != null ? `${fmt(hw.mem_used_mib / 1024, 1)}G` : '–';
  $('mnTemp').textContent = hw.temp_c != null ? `${fmt(hw.temp_c)}°C` : '–';
  $('mnPow').textContent = hw.power_w != null ? `${fmt(hw.power_w)}W` : '–';
  if (snap.gallery_count != null) $('cntGal').textContent = snap.gallery_count;
}

/* -------- Monitor Tabs & Logs -------- */
function switchMonTab(t_) {
  S.monTab = t_;
  document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.t === t_));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('on', p.id === 'p-' + t_));
  if (t_ === 'logs') fetchLogs();
}
document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => switchMonTab(b.dataset.t));
$('lgFilter').oninput = fetchLogs;

async function fetchLogs() {
  try {
    const r = await fetch('/api/logs?tail=300', { cache: 'no-store' });
    if (!r.ok) return;
    const { lines } = await r.json();
    const q = $('lgFilter').value.trim().toLowerCase();
    const rows = q ? lines.filter(l => (l.msg || '').toLowerCase().includes(q)) : lines;
    const body = $('lgBody');
    body.innerHTML = rows.length ? rows.map(l => {
      const m = l.msg || '';
      const cls = /fail|error|nan|exception|abort/i.test(m) ? 'err' : /\[g2e\] prompt |\[H3ImagePipe\]|\[Qwen21Pipe\]|\[H3Pipe\]/.test(m) ? 'req'
        : /listening on|ready in|Direct MME|serving /i.test(m) ? 'ok' : '';
      return `<div class="ln ${cls}"><span class="ts">${l.ts ? hms(l.ts) : ''}</span><span class="msg">${esc(m)}</span></div>`;
    }).join('') : `<div class="ln dim"><span class="msg">${esc(t('no_logs'))}</span></div>`;
    if ($('lgAuto').checked) body.scrollTop = body.scrollHeight;
  } catch (e) {}
}

/* -------- Polling & Theme -------- */
async function fetchSnap() {
  try {
    const r = await fetch('/api/snapshot', { cache: 'no-store' });
    if (!r.ok) return;
    const snap = await r.json();
    S.snap = snap;
    renderTopMiniHw(snap);
    renderEngineStrip(snap);
    if (S.view === 'monitor') renderMonitor(snap);
  } catch (e) {
    $('stDot').className = 'sdot err'; $('stText').textContent = t('disconnected');
  }
}
function schedule() {
  clearInterval(S.timer);
  S.timer = setInterval(() => {
    if (S.paused) return;
    fetchSnap();
    if (S.view === 'monitor' && S.monTab === 'logs') fetchLogs();
  }, S.interval);
}
document.querySelectorAll('#segInt button').forEach(b => b.onclick = () => {
  document.querySelectorAll('#segInt button').forEach(x => x.classList.toggle('on', x === b));
  S.interval = +b.dataset.v; schedule();
});
$('btnPause').onclick = () => { S.paused = !S.paused; $('btnPause').innerHTML = ico(S.paused ? 'play' : 'pause'); };
function applyTheme(th) {
  document.documentElement.setAttribute('data-theme', th);
  $('btnTheme').innerHTML = ico(th === 'dark' ? 'sun' : 'moon');
  localStorage.setItem('g2e-apple-theme', th);
  if (S.snap && S.view === 'monitor') renderMonitor(S.snap);
}
$('btnTheme').onclick = () => applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
let rz; window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { if (S.snap && S.view === 'monitor') renderMonitor(S.snap); }, 120); });

// Init
applyTheme(localStorage.getItem('g2e-apple-theme') || 'light');
applyLang(S.lang);
setPerf('live');
switchView(S.view);
fetchSnap();
loadGallery();
schedule();
