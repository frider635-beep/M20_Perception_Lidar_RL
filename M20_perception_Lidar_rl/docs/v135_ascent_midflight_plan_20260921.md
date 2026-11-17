# V135 中后段交替与全身支撑核查（方案，未改训练）

## 核实事实
- 回放模型32100；部署run20260921_123314_52212 manifest确认同一checkpoint，实际加载v11_signed_edges_v2；manifest中gru128字符串是launcher默认字段，不能据此判定加载GRU。
- 实际MuJoCo stair.xml踏面0.30m、阶高0.17m、6级（末级平台）。训练踏面0.30m、阶高范围0.05–0.23m，8×8m倒金字塔，当前ledger统计每段7级。当前运行源码与manifest哈希一致。
- URDF前后髋x=±0.3141m，髋距0.6282m；大/小腿主要纵向长度各0.25m。髋距/踏面≈2.094。不是可达性/力矩验收，只提示前腿B3附近后腿开始入阶可能形成支撑瓶颈。
- adjacent reward逐级ordered触发，cycle条件chain>=2，没有B3终止；整段成功要求flight_complete及episode_ordered_valid。
- 近期32902–33101完整200轮日志：12129 full episodes，B1=10930，B2=10172，B3=9635，fullchain=7363（60.71%），每段7级。是训练ledger统计，非32100冻结回放结果，也非物理独立真值。
- 未修风险：target-height pool匹配仍使用锁定lane±0.15m；新target高度未获取则completion和shaping关闭。需实际记录残差，不能直接归因PPO。
- 当前上楼专项potential只使用前轮clearance/cross/support；whole-body质量限下楼启用。上楼有通用姿态/接触项，但没有对应的整机支撑转移质量信用。

## 单一路线
1. 固定32100，先同几何/command/入口对照Isaac和MuJoCo。每级记录进入数、正确完成数、断链原因、target/height validity和scanner残差，直至实际末级；独立按真实轮接触落点校验ledger。记录四轮载荷和轮-髋相对位置、接触滑移、root滚转/俯仰/角速度、关节/力矩余量。单独标记首次后轮入阶。
2. 修physical edge matching：world-plane纵向身份+真实边线跨度+高度一致性，不把第一条扫描射线的横向lane当无限期锁定点。保持唯一physical ledger；不增加Actor输入；用侧移、yaw、真实skip反例验证。
3. 新增仅上楼的whole-body quality读数和一个有界正向信用；先测再确定阈值。利用现有前腿progress/ordered事件作为有限结算预算，结合支撑腿有效承重、髋轮工作空间、接触滑移和相对地形的机身姿态/角速度。禁止站着持续领bonus；不要求双后轮始终接地、不固定膝角、不罚自然地形俯仰、不用硬support-lock。保留基础速度驱动；支撑质量只调节小幅上楼额外收益，不直接砍全局速度奖励。
4. 识别单阶/短楼梯/长楼梯的训练分布缺口。先把ledger固定step_width推进改为可验证的真实相邻边，再添加级数1/3/6/10/12的上楼对照与训练覆盖；最初保持0.30m踏面。只有同模型几何扫描证明需要才拓宽踏面分布，不能改一处terrain却保留旧ledger步距。
5. 下楼、平地恢复代码/权重、Actor63 MLP、Critic574、PPO8192×24冻结。共享参数更新仍可能影响冻结行为，所以相同seeds/commands做下楼和平地回归。

## 验收
逐级条件成功率P(Bk|B1…B{k-1})与整段成功率，特别B3→B4和首次后轮入阶；速度/完成时间与基线对比，防止停走换成功。后腿承重/工作空间/滑移、倾斜与角速度异常下降；下楼和平地回归不退化。不能凭总reward、B3统计或训练ledger单一视角宣布解决。

本轮未运行新的物理回放，未修改reward/训练配置，未停止当前训练；具体动态因果待事件轨迹验证。
