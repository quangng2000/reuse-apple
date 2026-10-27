extension RealtimeStreamingEnhancer {
    /// Takes exclusive use of an existing enhancer on the caller's serial worker.
    /// Do not push/reset/finish the supplied enhancer elsewhere while wrapped.
    /// It is reset before the wrapper's first push; live recovery never calls finish().
    public convenience init(enhancer: StreamingEnhancer, configuration: RealtimeStreamingConfiguration) {
        self.init(configuration: configuration, push: enhancer.push, reset: enhancer.reset)
    }
}
