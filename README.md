# 同伴带动与交叉接触下的邀学试验决策

这是一个面向社区老年教育运营的本地分析原型，使用合成记录演示从业务记录到运营摘要的处理。项目把记录读取、统计口径和运营判断分开，便于核对原始事实与派生结果。

## 环境与使用

需要 Python 3.11 或更高版本，无第三方依赖。运行 `python3 -m unittest discover -s tests -v` 执行测试；运行 `python3 -m learning_peer_experiment.report` 查看合成数据摘要。所有记录均为本地合成数据，不包含个人信息。
也可使用 Docker 运行：`docker build -t learning-peer-experiment .`，再执行 `docker run --rm learning-peer-experiment`。容器内运行测试可执行 `docker run --rm --entrypoint python3 learning-peer-experiment -m unittest discover -s tests -v`。

## 现有行为

抽签设计决定暴露可识别性；效果证据决定可用社区增量；整体邀请计划必须尊重支持范围与重复收益。

当前摘要只反映样例记录的基础汇总，不用于形成个体资格或运营承诺。SQLite 数据库在调用时创建于临时目录，原始 CSV 保持不变。
