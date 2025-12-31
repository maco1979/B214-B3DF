declare class ApiGatewayServer {
    private app;
    private port;
    constructor(port?: number);
    private setupMiddlewares;
    private setupRoutes;
    private setupErrorHandling;
    start(): Promise<void>;
    stop(): Promise<void>;
}
export { ApiGatewayServer };
//# sourceMappingURL=server.d.ts.map