# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral guidelines to reduce common LLM coding mistakes.

用中文回答，在每个回答开始称呼"MayorG0"

## 1. 项目架构

```
core/          # 底层库 (不常改): heatmap, ga_allocator, spiral_search, dubins_planner, uav_navi
emergency/     # 主工作区: agent, simulator, utils, train, eval, demo, plot_training
docs/          # 文档区
```

**多步 MDP**: n 个受影响目标 → n 步序列决策。前 n-1 步 `done=False` (中间即时奖励), 第 n 步 `done=True` (终端全局奖励)。

## 2. 常用命令

```bash
venv/Scripts/python -m emergency.train --episodes 100000          # 从头训练
venv/Scripts/python -m emergency.train --episodes 100000 --resume auto  # 续训
venv/Scripts/python -m emergency.demo --type S1                   # 单场景演示 (S1/S2/S3/S4/random)
venv/Scripts/python -m emergency.eval --scenarios 100             # 5方法对比评估
venv/Scripts/python -m emergency.plot_training                    # 训练曲线
```

产物: `models/*.pt`, `checkpoints/*.pt`, `models/*_log.csv`, `outputs/emergency_demo_*.png`, `outputs/emergency_eval_*.png`

## 3. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 4. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 5. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 6. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
