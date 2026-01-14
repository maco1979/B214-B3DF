# Page snapshot

```yaml
- generic [ref=e4]:
  - generic [ref=e5]:
    - heading "赛博有机体" [level=3] [ref=e6]
    - paragraph [ref=e7]: 登录您的账户以继续
  - generic [ref=e9]:
    - generic [ref=e10]:
      - text: 邮箱
      - generic [ref=e11]:
        - img [ref=e12]
        - textbox "邮箱" [ref=e15]:
          - /placeholder: example@example.com
    - generic [ref=e16]:
      - generic [ref=e17]:
        - generic [ref=e18]: 密码
        - link "忘记密码？" [ref=e19]:
          - /url: "#"
      - generic [ref=e20]:
        - img [ref=e21]
        - textbox "密码" [ref=e24]:
          - /placeholder: ••••••••
    - button "登录" [ref=e25] [cursor=pointer]
    - generic [ref=e30]: 或者使用其他方式登录
    - generic [ref=e31]:
      - button "扫码登录" [ref=e32] [cursor=pointer]:
        - img [ref=e33]
        - text: 扫码登录
      - button "GitHub登录" [ref=e36] [cursor=pointer]:
        - img [ref=e37]
        - text: GitHub登录
    - button "没有账号？使用产品注册码注册" [ref=e41] [cursor=pointer]
    - generic [ref=e43]: "测试账号: test@example.com / 密码: test123456"
```