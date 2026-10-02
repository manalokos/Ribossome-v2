//! Device, queue e utilitários de GPU.

pub mod profiler;

pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Gpu {
    /// Cria o device. Com `surface`, escolhe um adaptador que consiga
    /// apresentar nela; sem, serve para testes sem janela.
    pub async fn new(instance: wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<Self, String> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: surface,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| format!("sem adaptador GPU: {e}"))?;
        let info = adapter.get_info();
        log::info!("GPU: {} ({:?}, {:?})", info.name, info.backend, info.device_type);

        // Pedimos os limites do adaptador: a grelha de 2048² ocupa 64 MB.
        // Timestamps dentro dos passes (medição por kernel), se houver.
        let ts = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        let features = adapter.features() & ts;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ribossome"),
                required_features: features,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("request_device falhou: {e}"))?;
        Ok(Self { instance, adapter, device, queue })
    }

    pub fn new_headless() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        pollster::block_on(Self::new(instance, None))
    }

    /// Espera que todo o trabalho submetido termine.
    pub fn wait_idle(&self) {
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll(Wait) falhou");
    }

    /// Lê um buffer inteiro de forma síncrona (só para testes e depuração).
    pub fn read_buffer_blocking(&self, src: &wgpu::Buffer) -> Vec<u8> {
        let size = src.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        staging.map_async(wgpu::MapMode::Read, .., |r| r.expect("map_async falhou"));
        self.wait_idle();
        let data = staging.get_mapped_range(..).expect("get_mapped_range").to_vec();
        staging.unmap();
        data
    }
}

/// Número de workgroups para cobrir `n` com grupos de `wg`.
pub fn groups(n: u32, wg: u32) -> u32 {
    n.div_ceil(wg)
}
