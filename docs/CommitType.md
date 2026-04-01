| Type         | 定义                            | 使用场景                   | 示例                                   | 边界说明                   | 常见错误               |
| ------------ | ------------------------------- | -------------------------- | -------------------------------------- | -------------------------- | ---------------------- |
| **feat**     | 新功能，引入新的系统能力        | 新接口、新页面、新业务逻辑 | `feat(auth): add jwt login`            | 用户可以做“之前做不到的事” | ❌ 用于重构或优化       |
| **fix**      | 修复 bug 或错误行为             | 逻辑错误、崩溃、异常处理   | `fix(order): prevent duplicate submit` | 修复“已有功能的不正确行为” | ❌ 把优化写成 fix       |
| **docs**     | 文档相关修改                    | README、API文档、注释      | `docs(api): update auth guide`         | 不涉及代码逻辑变化         | ❌ 修改逻辑却写 docs    |
| **style**    | 代码格式/风格调整（无逻辑变化） | 格式化、缩进、命名微调     | `style: format with prettier`          | 不影响运行结果             | ❌ 修改逻辑却用 style   |
| **refactor** | 重构代码（不改变功能）          | 代码拆分、结构优化、去重复 | `refactor(auth): extract token logic`  | 输入输出完全一致           | ❌ 修 bug 写成 refactor |
| **perf**     | 性能优化                        | SQL优化、缓存、算法优化    | `perf(api): reduce latency`            | 明确提升性能指标           | ❌ 普通重构写 perf      |
| **test**     | 测试相关                        | 单元测试、集成测试、mock   | `test(user): add login test`           | 仅测试代码变化             | ❌ 修改业务代码写 test  |
| **chore**    | 杂项/非业务代码                 | 配置、脚本、清理代码       | `chore: update eslint config`          | 不影响业务逻辑             | ❌ 滥用为“兜底类型”     |
| **build**    | 构建系统或依赖构建              | webpack、vite、docker      | `build: update dockerfile`             | 影响打包/构建流程          | ❌ CI 修改写 build      |
| **ci**       | CI/CD流程                       | GitHub Actions、Jenkins    | `ci: add commitlint check`             | 影响自动化流程             | ❌ 构建配置写 ci        |
| **revert**   | 回滚提交                        | 撤销历史 commit            | `revert: feat(auth): add jwt login`    | 必须是“撤销行为”           | ❌ 手动写错 revert      |