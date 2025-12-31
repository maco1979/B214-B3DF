export const metricsMiddleware = (req, res, next) => {
    const startTime = Date.now();
    res.on('finish', () => {
        const duration = Date.now() - startTime;
        const statusCode = res.statusCode;
        console.log(`METRICS: ${req.method} ${req.path} - ${statusCode} - ${duration}ms`);
    });
    next();
};
//# sourceMappingURL=metrics.js.map