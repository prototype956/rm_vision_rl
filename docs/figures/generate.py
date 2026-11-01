"""Generate documentation diagrams as SVG and editable Mermaid from local specs.

Run: python docs/figures/generate.py
Only the Python standard library is required. No training or network access.
"""
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parent
COLORS = {'control': ('#eef5ff', '#4475ad'), 'train': ('#f0ecfa', '#8465ac'),
          'check': ('#eaf5f0', '#3d8066'), 'warn': ('#fff4e5', '#b67a28'),
          'stop': ('#fbeeee', '#ad5454'), 'neutral': ('#f3f5f8', '#748296')}


class Diagram:
    def __init__(self, name, title, subtitle, width, height):
        self.name, self.width, self.height = name, width, height
        self.nodes, self.edges = {}, []
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#61758a"/></marker></defs>',
            '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#25394c} .edge{fill:none;stroke:#61758a;stroke-width:2;marker-end:url(#arrow)} .label{font-size:16px;paint-order:stroke;stroke:white;stroke-width:7;stroke-linejoin:round}</style>',
            f'<rect width="{width}" height="{height}" fill="white"/>']
        self.text(35, 40, title, 25, bold=True)
        self.text(35, 72, subtitle, 16)

    def text(self, x, y, value, size=18, bold=False, anchor='start', color=None):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"'+
            (' font-weight="600"' if bold else '')+(f' style="fill:{color}"' if color else '')+f'>{escape(value)}</text>')

    def node(self, key, x, y, w, h, lines, kind='control'):
        self.nodes[key] = (x, y, w, h, lines, kind)

    def edge(self, a, b, points, label='', at=None, dashed=False):
        self.edges.append((a, b, points, label, at, dashed))

    def finish(self, notes=()):
        for a, b, points, label, at, dashed in self.edges:
            assert a in self.nodes and b in self.nodes
            path='M'+' L'.join(f'{x},{y}' for x,y in points)
            self.parts.append(f'<path class="edge" d="{path}"'+(' stroke-dasharray="6 5"' if dashed else '')+'/>')
            if label:
                x,y=at
                self.parts.append(f'<text class="label" x="{x}" y="{y}" text-anchor="middle">{escape(label)}</text>')
        for key,(x,y,w,h,lines,kind) in self.nodes.items():
            fill,stroke=COLORS[kind]
            self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
            start=y+h/2-(len(lines)-1)*13+6
            for i,line in enumerate(lines):
                self.text(x+w/2,start+i*26,line,18 if i==0 else 16,i==0,'middle')
        for y,line in notes:self.text(35,y,line,16)
        self.parts.append('</svg>')
        (ROOT/(self.name+'.svg')).write_text('\n'.join(self.parts)+'\n',encoding='utf-8')
        m=['flowchart TB']
        for key,(_,_,_,_,lines,kind) in self.nodes.items():
            label='<br/>'.join(lines).replace('"','&quot;')
            m.append(f'  {key}["{label}"]:::{kind}')
        for a,b,_,label,_,dash in self.edges:
            arrow='-.->' if dash else '-->'
            m.append(f'  {a} {arrow}'+(f'|"{label}"|' if label else '')+f' {b}')
        for kind,(fill,stroke) in COLORS.items():m.append(f'  classDef {kind} fill:{fill},stroke:{stroke},color:#25394c')
        (ROOT/(self.name+'.mmd')).write_text('\n'.join(m)+'\n',encoding='utf-8')


def architecture():
    d=Diagram('control_loop','闭环数据流与训练信息边界','实线：控制周期与观测；虚线：仅训练 / 独立评估通道。每次 Advance 推进 10 ms。',1200,820)
    d.node('sim',40,150,300,100,['Rust 同步仿真','物理 / 测量 / 自身反馈'])
    d.node('vision',450,150,300,100,['C++ 视觉估计','PnP → ArmorPredictor'])
    d.node('control',860,150,300,100,['C++ 火控与执行','弹道 → TinyMPC → 命令'])
    d.node('adapter',450,350,300,100,['Python 观测适配','v5 白名单 / 掩码 / 时钟'])
    d.node('actor',860,350,300,100,['GRU Actor','本地记忆 → 选板 / 请求开火'])
    d.node('score',40,590,300,105,['独立评估与奖励统计','事件 / 实际出膛 / 伤害'],'train')
    d.node('critic',450,590,300,105,['独立 GRU Critic','合法观测 + 当前 privileged'],'train')
    d.node('ppo',860,590,300,105,['完整回合循环 PPO','分别优化 Actor 与 Critic'],'train')
    d.edge('sim','vision',[(340,200),(450,200)],'角点 / 反馈',(395,180))
    d.edge('vision','adapter',[(600,250),(600,350)],'估计与候选',(665,305))
    d.edge('adapter','actor',[(750,400),(860,400)],'Actor 输入',(805,380))
    d.edge('actor','control',[(1010,350),(1010,250)],'九动作',(1060,305))
    d.edge('control','sim',[(1010,150),(1010,110),(190,110),(190,150)],'Python 提交 Advance，推进后 ack',(600,101))
    d.edge('sim','score',[(190,250),(190,590)],'事件 / 结果',(250,500),True)
    d.edge('sim','critic',[(340,230),(385,230),(385,642),(450,642)],'仅训练真值',(390,560),True)
    d.edge('adapter','critic',[(600,450),(600,590)],'合法观测',(665,525),True)
    d.edge('critic','ppo',[(750,642),(860,642)],'价值',(805,622),True)
    d.edge('actor','ppo',[(1125,450),(1125,590)],'轨迹 / 行为概率',(1080,525),True)
    d.edge('score','ppo',[(190,695),(190,738),(1010,738),(1010,695)],'训练：逐步奖励；评估尾部伤害不追加到 PPO',(600,730),True)
    d.finish([(785,'Actor 只读取白名单；真值、实际伤害与事件账本不作为 Actor 输入。NUC 仅部署 Actor 与本地状态。')])


def memory():
    d=Diagram('gru_lifecycle','GRU 记忆的重置、保留与置换','规则按每个环境独立执行；“清零”在该真实控制步的 GRU 更新之前发生。',1200,830)
    rows=[
        ('episode','新回合 episode_start',['目标 / 候选 / 云台 / 资源 / Critic 全部清零'],'warn'),
        ('target','新代次 / tracker reset / 首次进入 LOST',['目标与候选清零；云台 / 资源 / Critic 保留后更新'],'warn'),
        ('coast','短暂遮挡，仍未进入 LOST',['全部状态继续更新；是否可开火由掩码决定'],'check'),
        ('map','可信 candidate_map 重排',['候选记忆按新槽位 → 旧槽位置换；其余状态保留后更新'],'control'),
        ('badmap','candidate_map 无效',['目标与候选清零，清本地返回槽位；其余状态保留后更新'],'warn'),
        ('padding','序列填充 active = false',['所有状态保持原值：不清零、不置换、不推进'],'neutral')]
    for i,(key,event,result,kind) in enumerate(rows):
        y=115+i*95
        d.node(key,35,y,385,68,[event],kind)
        d.node(key+'state',510,y,650,68,result,kind)
        d.edge(key,key+'state',[(420,y+34),(510,y+34)])
    d.finish([(730,'Actor 状态：target [B,128]；candidates [B,4,64]；gimbal / resources 各 [B,64]。'),
        (760,'Critic 独立保存 [B,128]；target_reset 不清 Critic。candidate_map 不使用仿真真实装甲 ID。'),
        (790,'回合首步 dt_s = 0；后续真实控制步推进记忆。部署换权重只在下一回合首步发生。')])


def v2():
    d=Diagram('curriculum_v2_release','课程 v2：预检、资格验证与冻结训练','此图展开长期冻结流水线；单独的 curriculum_v2 run 不等于完成全部流水线门槛。',1160,890)
    specs=[('pre',115,['preflight：固定验证 / 测试场景','保留准入失败与种子，不按战斗成绩换场景']),
        ('qual',235,['benchmark：两类角色各完成至少两次更新','128 环境，检查恢复计数与 Actor 导出']),
        ('gate',355,['长期流水线核查回归与放行文件','缺测 / 物理 / 规则诊断；源码身份与测试']),
        ('freeze',475,['冻结源码、配置、二进制、资产与 Python 环境','记录哈希与运行目录，从冻结副本启动']),
        ('train',595,['正式训练：步兵 → 哨兵，各自从零开始','每角色累计十亿步；每次更新后验证 / 复核']),
        ('final',715,['独立最终测试、Actor 导出与一致性核对','分别报告：预算完成 / 战斗门槛 / 热锁门槛'])]
    for key,y,lines in specs:d.node(key,45,y,730,82,lines,'train' if key=='train' else 'check')
    for (a,y,_),(b,z,_) in zip(specs,specs[1:]):d.edge(a,b,[(410,y+82),(410,z)])
    d.node('blocked',850,355,270,120,['核查失败','保留失败与诊断产物','阻止正式启动'],'stop')
    d.edge('gate','blocked',[(775,396),(850,396)],'失败',(812,380))
    d.finish([(845,'资格验证步数不计入正式预算。检查点恢复使用原累计预算、环境数与课程，从新回合零隐藏状态开始。')])


def v3():
    d=Diagram('curriculum_v3_release','课程 v3 修订 3：校准、迁移与正式发布','适用于当前“步兵 v2 检查点迁移，哨兵从零训练”的发布入口。箭头表示证据依赖。',1200,1100)
    specs=[('pre',115,['preflight：全路径可见性与物理准入','冻结场景和证书，绑定原生二进制 / 配置身份']),
        ('cal',245,['calibrate：传统火控校准 → 冻结门槛 → 独立验证','T1/T2：每车型 × 运动组合分别检查；测试集不调参']),
        ('migration',375,['显式迁移 v2 → v3，并隔离复评旧已解锁阶段','recurrent.pt 字节不变，清旧晋级历史，定位未通过阶段']),
        ('qual',505,['benchmark：两角色各完成两次新增更新','检查恢复、Actor 导出及配置 / 场景 / 校准身份']),
        ('gate',635,['run 发布核查：资格验证 + 原始 / 修复碰撞回归','核对检查点、场景、校准、配置与二进制哈希']),
        ('freeze',765,['复制并校验独立运行副本，重定位检查点路径','正式任务仍从原迁移检查点出发']),
        ('train',895,['正式训练：恢复步兵 → 从零训练哨兵','累计十亿步目标；最终测试与预算 / 战斗 / 热锁分别报告'])]
    for key,y,lines in specs:d.node(key,35,y,800,86,lines,'train' if key=='train' else 'check')
    for (a,y,_),(b,z,_) in zip(specs,specs[1:]):d.edge(a,b,[(435,y+86),(435,z)])
    d.node('collision',900,505,270,86,['碰撞回归报告','原错误复现 + 修复通过'],'check')
    d.edge('collision','gate',[(1035,591),(1035,678),(835,678)])
    d.node('blocked',900,765,270,110,['任何发布核查失败','保留失败与诊断','阻止正式启动'],'stop')
    d.edge('gate','blocked',[(835,700),(875,700),(875,820),(900,820)])
    d.finish([(1030,'资格验证优化结果不作为正式训练起点；校准、验证和测试种子分离。旧修订证书与基线不能替代修订 3 证据。'),
        (1060,'固定案例 MCAP 可在资格验证后显式录制；它用于诊断，不替代发布核查，也不按命中成绩选案例。')])


if __name__ == '__main__':
    for build in (architecture,memory,v2,v3):build()
    print('Generated four SVG diagrams and four Mermaid sources.')
