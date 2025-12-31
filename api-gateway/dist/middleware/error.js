export const errorHandler = (err, _req, res, _next) => {
    console.error('API网关错误:', err);
    if (err.code === 'ECONNREFUSED') {
        res.status(503).json({
            error: '服务不可用',
            message: '后端服务暂时不可用，请稍后重试'
        });
        return;
    }
    if (err.response && err.response.status) {
        res.status(err.response.status).json({
            error: '后端服务错误',
            message: err.response.data?.message || '后端服务处理错误'
        });
        return;
    }
    res.status(500).json({
        error: '内部服务器错误',
        message: '服务器遇到意外错误'
    });
};
//# sourceMappingURL=error.js.map