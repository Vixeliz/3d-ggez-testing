//! How to draw a 3D cube in ggez.
//!
//! ggez doesn't provide any 3D drawing itself, but it exposes
//! the underlying `gfx-rs` data types, so you can bypass ggez's
//! drawing code entirely and write your own.

use crevice::std140::AsStd140;
use ggez::event;
use ggez::glam::*;
use ggez::graphics;
use ggez::graphics::Color;
use ggez::graphics::Rect;
use ggez::{Context, GameResult};
use std::env;
use std::f32;
use std::path;
use wgpu::util::DeviceExt;

type Isometry3 = Mat4;
type Point3 = Vec3;
type Vector3 = Vec3;

struct Camera {
    eye: Vector3,
    target: Vector3,
    up: Vector3,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            // position the camera one unit up and 2 units back
            // +z is out of the screen
            eye: (0.0, 1.0, 2.0).into(),
            // have it look at the origin
            target: (0.0, 0.0, 0.0).into(),
            // which way is "up"
            up: Vector3::Y,
            aspect: 16.0 / 9.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        }
    }
}

impl Camera {
    fn build_view_projection_matrix(&self) -> Mat4 {
        // 1.
        let view = default_view();
        // 2.
        let proj = Mat4::perspective_rh(f32::consts::PI / 4.0, 4.0 / 3.0, 1.0, 10.0);

        // 3.
        return proj * view;
        // OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}

#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // We can't use cgmath with bytemuck directly so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: [
                Mat4::IDENTITY.x_axis.into(),
                Mat4::IDENTITY.y_axis.into(),
                Mat4::IDENTITY.z_axis.into(),
                Mat4::IDENTITY.w_axis.into(),
            ],
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        let view = camera.build_view_projection_matrix();
        self.view_proj = [
            view.x_axis.into(),
            view.y_axis.into(),
            view.z_axis.into(),
            view.w_axis.into(),
        ];
    }
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Vertex {
    pos: [f32; 4],
    tex_coord: [f32; 2],
}

impl Vertex {
    fn new(p: [i8; 3], t: [i8; 2]) -> Vertex {
        Vertex {
            pos: [f32::from(p[0]), f32::from(p[1]), f32::from(p[2]), 1.0],
            tex_coord: [f32::from(t[0]), f32::from(t[1])],
        }
    }
}

fn default_view() -> Isometry3 {
    // Eye location, target location, up-vector
    Mat4::look_at_rh(
        Point3::new(1.5f32, -5.0, 3.0),
        Point3::new(0f32, 0.0, 0.0),
        Vector3::Z,
    )
}

struct MainState {
    frames: usize,
    camera: Camera,
    screen_coords: Rect,

    verts: wgpu::Buffer,
    inds: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    depth: graphics::ScreenImage,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

impl MainState {
    fn new(ctx: &mut Context) -> GameResult<Self> {
        // Shaders.
        let shader = ctx
            .gfx
            .wgpu()
            .device
            .create_shader_module(wgpu::include_wgsl!("../resources/cube.wgsl"));

        // Cube geometry
        let vertex_data = [
            // top (0, 0, 1)
            Vertex::new([-1, -1, 1], [0, 0]),
            Vertex::new([1, -1, 1], [1, 0]),
            Vertex::new([1, 1, 1], [1, 1]),
            Vertex::new([-1, 1, 1], [0, 1]),
            // bottom (0, 0, -1)
            Vertex::new([-1, 1, -1], [1, 0]),
            Vertex::new([1, 1, -1], [0, 0]),
            Vertex::new([1, -1, -1], [0, 1]),
            Vertex::new([-1, -1, -1], [1, 1]),
            // right (1, 0, 0)
            Vertex::new([1, -1, -1], [0, 0]),
            Vertex::new([1, 1, -1], [1, 0]),
            Vertex::new([1, 1, 1], [1, 1]),
            Vertex::new([1, -1, 1], [0, 1]),
            // left (-1, 0, 0)
            Vertex::new([-1, -1, 1], [1, 0]),
            Vertex::new([-1, 1, 1], [0, 0]),
            Vertex::new([-1, 1, -1], [0, 1]),
            Vertex::new([-1, -1, -1], [1, 1]),
            // front (0, 1, 0)
            Vertex::new([1, 1, -1], [1, 0]),
            Vertex::new([-1, 1, -1], [0, 0]),
            Vertex::new([-1, 1, 1], [0, 1]),
            Vertex::new([1, 1, 1], [1, 1]),
            // back (0, -1, 0)
            Vertex::new([1, -1, 1], [0, 0]),
            Vertex::new([-1, -1, 1], [1, 0]),
            Vertex::new([-1, -1, -1], [1, 1]),
            Vertex::new([1, -1, -1], [0, 1]),
        ];

        #[rustfmt::skip]
        let index_data: &[u32] = &[
             0,  1,  2,  2,  3,  0, // top
             4,  5,  6,  6,  7,  4, // bottom
             8,  9, 10, 10, 11,  8, // right
            12, 13, 14, 14, 15, 12, // left
            16, 17, 18, 18, 19, 16, // front
            20, 21, 22, 22, 23, 20, // back
        ];

        // Create vertex and index buffers.
        let verts = ctx
            .gfx
            .wgpu()
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(vertex_data.as_slice()),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let inds = ctx
            .gfx
            .wgpu()
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(index_data),
                usage: wgpu::BufferUsages::INDEX,
            });

        let camera = Camera::default();
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer =
            ctx.gfx
                .wgpu()
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Camera Buffer"),
                    contents: bytemuck::cast_slice(&[camera_uniform]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
        let camera_bind_group_layout =
            ctx.gfx
                .wgpu()
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                    label: Some("camera_bind_group_layout"),
                });
        let texture_bind_group_layout =
            ctx.gfx
                .wgpu()
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                    label: Some("texture_bind_group_layout"),
                });
        let camera_bind_group =
            ctx.gfx
                .wgpu()
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &camera_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    }],
                    label: Some("camera_bind_group"),
                });

        let render_pipeline_layout =
            ctx.gfx
                .wgpu()
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline =
            ctx.gfx
                .wgpu()
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vs_main",
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<Vertex>() as _,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[
                                // pos
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x4,
                                    offset: 0,
                                    shader_location: 0,
                                },
                                // tex_coord
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x2,
                                    offset: 16,
                                    shader_location: 1,
                                },
                            ],
                        }],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        unclipped_depth: false,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Greater,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: ctx.gfx.surface_format(),
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview: None,
                });

        // Create 1-pixel blue texture.
        let image =
            graphics::Image::from_solid(ctx, 1, graphics::Color::from_rgb(0x20, 0xA0, 0xC0));

        let sampler = ctx
            .gfx
            .wgpu()
            .device
            .create_sampler(&graphics::Sampler::default().into());

        let bind_group = ctx
            .gfx
            .wgpu()
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(image.wgpu().1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

        let depth = graphics::ScreenImage::new(ctx, graphics::ImageFormat::Depth32Float, 1., 1., 1);

        // FOV, spect ratio, znear, zfar
        // let proj = Mat4::perspective_rh(f32::consts::PI / 4.0, 4.0 / 3.0, 1.0, 10.0);
        // let transform = proj * default_view();

        Ok(MainState {
            frames: 0,
            camera: Camera::default(),
            verts,
            inds,
            pipeline,
            bind_group,
            depth,
            screen_coords: Rect {
                x: 0.,
                y: 0.,
                w: ctx.gfx.drawable_size().0,
                h: ctx.gfx.drawable_size().1,
            },
            camera_uniform,
            camera_buffer,
            camera_bind_group,
        })
    }

    // pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
    //     if new_size.width > 0 && new_size.height > 0 {
    //         self.size = new_size;
    //         self.config.width = new_size.width;
    //         self.config.height = new_size.height;
    //         self.surface.configure(&self.device, &self.config);
    //     }
    // }
}

impl event::EventHandler<ggez::GameError> for MainState {
    fn resize_event(&mut self, _ctx: &mut Context, width: f32, height: f32) -> GameResult {
        // println!("Resized screen to {}, {}", width, height);
        let new_rect = graphics::Rect::new(0.0, 0.0, width as f32, height as f32);
        self.screen_coords = new_rect;
        Ok(())
    }

    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, Color::BLACK);
        canvas.set_screen_coordinates(self.screen_coords);
        {
            let depth = self.depth.image(ctx);

            let frame = ctx.gfx.frame().clone();
            let cmd = ctx.gfx.commands().unwrap();
            let mut pass = cmd.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.wgpu().1,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(
                            graphics::LinearColor::from(graphics::Color::new(0.1, 0.1, 0.1, 1.0))
                                .into(),
                        ),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth.wgpu().1,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.),
                        store: false,
                    }),
                    stencil_ops: None,
                }),
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            // NEW!
            pass.set_bind_group(1, &self.camera_bind_group, &[]);
            // pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            // pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(0, self.verts.slice(..));
            pass.set_index_buffer(self.inds.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..36, 0, 0..1);
        }

        let mut canvas = graphics::Canvas::from_frame(ctx, None);

        // Do ggez drawing
        let dest_point1 = Vec2::new(10.0, 210.0);
        let dest_point2 = Vec2::new(10.0, 250.0);
        canvas.draw(
            &graphics::Text::new("You can mix ggez and wgpu drawing;"),
            dest_point1,
        );
        canvas.draw(
            &graphics::Text::new("it basically draws wgpu stuff first, then ggez"),
            dest_point2,
        );

        canvas.finish(ctx)?;

        // self.frames += 1;
        // if (self.frames % 10) == 0 {
        //     println!("FPS: {}", ctx.time.fps());
        // }

        Ok(())
    }
}

pub fn main() -> GameResult {
    let resource_dir = if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let mut path = path::PathBuf::from(manifest_dir);
        path.push("resources");
        path
    } else {
        path::PathBuf::from("./resources")
    };

    let cb = ggez::ContextBuilder::new("cube", "ggez")
        .window_mode(ggez::conf::WindowMode::default().resizable(true))
        .add_resource_path(resource_dir);

    let (mut ctx, events_loop) = cb.build()?;
    let state = MainState::new(&mut ctx)?;
    event::run(ctx, events_loop, state)
}
