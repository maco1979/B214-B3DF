export const authMiddleware = (req, res, next) => {
    if (req.method === 'OPTIONS') {
        return next();
    }
    const apiKey = req.headers['x-api-key'];
    if (!apiKey) {
        res.status(401).json({
            error: '未提供API密钥',
            message: '请在请求头中提供有效的X-API-KEY'
        });
        return;
    }
    const validApiKeys = ['your-api-key-here', 'default-api-key'];
    if (!validApiKeys.includes(apiKey)) {
        res.status(403).json({
            error: '无效的API密钥',
            message: '提供的API密钥无效'
        });
        return;
    }
    next();
};
//# sourceMappingURL=auth.js.map