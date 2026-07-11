const PHYSICS_BLOB_COUNT = 3u;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

struct SimParams {
  time: f32,
  dt: f32,
  resolution: vec2f,
  mouse: vec2f,
};

struct Blob {
  pos: vec2f,
  vel: vec2f,
};

@group(0) @binding(0) var<uniform> params: SimParams;

// ---- Compute stage: advances the two buoyant blobs' physics on the
// GPU. Runs once per frame, before the render pass reads the result. ----

@group(0) @binding(1) var<storage, read_write> blobsRW: array<Blob>;

// A pseudo-random hash: not truly random, but scattered enough that
// nearby inputs give unrelated outputs -- the building block every
// noise function starts from.
fn hash21(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

// Value noise: hash the four corners of the grid cell containing p,
// then smoothly interpolate between them. Unlike hash21 alone (which
// jumps randomly pixel to pixel), this varies gradually -- the
// "smooth randomness" that makes motion look organic instead of jittery.
fn noise2D(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  let u = f * f * (3.0 - 2.0 * f); // ease the interpolation, avoids visible grid seams
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

@compute @workgroup_size(PHYSICS_BLOB_COUNT)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  var b = blobsRW[i];
  let fi = f32(i);

  // Heat field: slow-evolving noise sampled purely in time (each blob
  // offset so they don't move in lockstep). Deliberately *not* a
  // function of the blob's own position -- coupling heat to pos.x
  // creates a feedback loop (move -> heat changes -> accelerate ->
  // move faster) that reads as jittery instead of calm.
  let heat = noise2D(vec2f(fi * 5.0, params.time * 0.08)) * 2.0 - 1.0;

  // A gentle spring pulling back toward the vertical center. Unlike a
  // constant gravity term, this is proportional to displacement -- the
  // further a blob drifts from the middle, the harder it's pulled
  // back -- so blobs settle into hovering near center instead of
  // drifting to a wall and waiting to bounce off it.
  let restoreStrength = 0.4;
  let heatStrength = 0.6;
  var ay = (0.0 - b.pos.y) * restoreStrength + heat * heatStrength;
  var ax = 0.0;

  // Mild repulsion so blobs don't sit directly on top of each other.
  for (var j = 0u; j < PHYSICS_BLOB_COUNT; j++) {
    if (j == i) {
      continue;
    }
    let other = blobsRW[j].pos;
    let delta = b.pos - other;
    let dist = max(length(delta), 0.001);
    let minDist = 0.22;
    if (dist < minDist) {
      let push = (minDist - dist) * 2.0;
      ax += (delta.x / dist) * push;
      ay += (delta.y / dist) * push;
    }
  }

  b.vel.x += ax * params.dt;
  b.vel.y += ay * params.dt;

  // Exponential velocity decay (viscous drag) instead of a subtractive
  // acceleration term: this is what actually makes the motion feel
  // slow and syrupy rather than springy. exp(-rate * dt) decays the
  // same amount per second regardless of frame rate.
  b.vel *= exp(-1.2 * params.dt);

  b.pos += b.vel * params.dt;

  // Keep blobs inside a rough container, losing a bit of energy on bounce.
  let boundX = 0.42;
  let boundY = 1.0;
  if (abs(b.pos.x) > boundX) {
    b.pos.x = clamp(b.pos.x, -boundX, boundX);
    b.vel.x *= -0.4;
  }
  if (abs(b.pos.y) > boundY) {
    b.pos.y = clamp(b.pos.y, -boundY, boundY);
    b.vel.y *= -0.4;
  }

  blobsRW[i] = b;
}

// ---- Render stage. Steps 10-11 built up 3D camera + raymarching from
// scratch; Step 13 reconnects the compute-simulated blobs above. ----

@group(0) @binding(1) var<storage, read> blobsRO: array<Blob>;

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );

  var out: VertexOutput;
  out.position = vec4f(pos[vertexIndex], 0.0, 1.0);
  out.uv = pos[vertexIndex] * 0.5 + 0.5;
  return out;
}

// Signed distance to a sphere: the 3D counterpart of Step 5's
// sdCircle, same idea -- negative inside, zero on the surface,
// positive outside.
fn sdSphere(p: vec3f, radius: f32) -> f32 {
  return length(p) - radius;
}

// Smooth minimum: identical to Step 6's 2D version -- smin operates
// on plain scalar distances, so it doesn't care whether those
// distances came from a 2D or 3D SDF. Blending multiple sdSphere
// calls with this is exactly what makes them merge into lava blobs
// instead of just overlapping.
fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

// The three blobs' X/Y positions now come straight from the compute
// shader's storage buffer -- the exact same buoyancy/repulsion
// simulation from Step 9, just read into a 3D scene instead of a 2D
// one. The simulation itself is still only 2D (it never touches a Z
// axis), so each blob gets a fixed Z offset here purely to spread
// them out in depth; X and Y are simulated, Z is not.
fn sceneSDF(p: vec3f) -> f32 {
  var zOffsets = array<f32, 3>(0.0, 0.35, -0.5);
  var radii = array<f32, 3>(0.4, 0.3, 0.35);

  let k = 0.4;
  var d = 1e5;
  for (var i = 0u; i < PHYSICS_BLOB_COUNT; i++) {
    let simPos = blobsRO[i].pos;
    let blobPos = vec3f(simPos.x, simPos.y, zOffsets[i]);
    let bd = sdSphere(p - blobPos, radii[i]);
    d = smin(d, bd, k);
  }
  return d;
}

// Surface normal via the SDF's gradient: nudge p a tiny amount along
// each axis and see how much the distance changes. The direction of
// steepest increase in distance points straight out of the surface --
// exactly the normal. This works for *any* SDF, however complex,
// without needing per-shape normal formulas.
fn estimateNormal(p: vec3f) -> vec3f {
  let e = 0.001;
  return normalize(vec3f(
    sceneSDF(p + vec3f(e, 0.0, 0.0)) - sceneSDF(p - vec3f(e, 0.0, 0.0)),
    sceneSDF(p + vec3f(0.0, e, 0.0)) - sceneSDF(p - vec3f(0.0, e, 0.0)),
    sceneSDF(p + vec3f(0.0, 0.0, e)) - sceneSDF(p - vec3f(0.0, 0.0, e)),
  ));
}

// Sphere tracing: walk along the ray in steps sized by the SDF's own
// output. Since sceneSDF(p) is the distance to the *nearest* surface
// in any direction, it's always safe to advance the ray by exactly
// that much without risk of stepping through a surface. Near a
// surface the steps shrink automatically; far away they leap ahead --
// no fixed step size needed.
fn raymarch(ro: vec3f, rd: vec3f) -> f32 {
  var t = 0.0;
  for (var i = 0; i < 100; i++) {
    let p = ro + rd * t;
    let d = sceneSDF(p);
    if (d < 0.001) {
      return t;
    }
    t += d;
    if (t > 50.0) {
      break;
    }
  }
  return -1.0; // no surface found within range: a miss
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let aspect = params.resolution.x / params.resolution.y;

  // Centered, aspect-corrected screen coordinate in roughly -1..1 --
  // same idea as Step 5's `p`, just feeding a 3D camera instead of an
  // SDF directly.
  var screen = in.uv * 2.0 - 1.0;
  screen.x *= aspect;

  // A slowly orbiting camera looking at the origin. Orbiting (rather
  // than a fixed camera) is a deliberate sanity check: if the ray
  // directions are really 3D perspective and not some flat reskin,
  // the resulting image will visibly rotate in a way a 2D effect
  // couldn't fake.
  let camDist = 3.0;
  let camPos = vec3f(
    sin(params.time * 0.3) * camDist,
    1.0,
    cos(params.time * 0.3) * camDist,
  );
  let lookTarget = vec3f(0.0, 0.0, 0.0);
  let worldUp = vec3f(0.0, 1.0, 0.0);

  // Camera basis: three mutually perpendicular directions describing
  // the camera's orientation, built purely from where it is and what
  // it's looking at -- this replaces a traditional view matrix.
  let forward = normalize(lookTarget - camPos);
  let right = normalize(cross(forward, worldUp));
  let up = cross(right, forward);

  // Field of view controls how much the screen's -1..1 extent spreads
  // the ray directions apart; a wider fov = a more "fisheye" spread.
  let fovRadians = radians(60.0);
  let tanHalfFov = tan(fovRadians * 0.5);

  let rayDir = normalize(
    forward + screen.x * tanHalfFov * right + screen.y * tanHalfFov * up,
  );

  let t = raymarch(camPos, rayDir);

  if (t < 0.0) {
    // Miss: a simple vertical sky gradient instead of flat black, so
    // there's still something to look at around the sphere.
    let skyT = rayDir.y * 0.5 + 0.5;
    let sky = mix(vec3f(0.02, 0.02, 0.05), vec3f(0.1, 0.12, 0.2), skyT);
    return vec4f(sky, 1.0);
  }

  let hitPoint = camPos + rayDir * t;
  let normal = estimateNormal(hitPoint);

  let lightPos = vec3f(2.0, 3.0, 2.0);
  let lightDir = normalize(lightPos - hitPoint);
  let viewDir = normalize(camPos - hitPoint);

  // Diffuse (Lambertian): brightness proportional to how directly the
  // surface faces the light. dot(normal, lightDir) is 1.0 head-on, 0
  // at a glancing angle, negative facing away -- clamped to 0 so it
  // never goes "negative bright".
  let diffuse = max(dot(normal, lightDir), 0.0);

  // Specular (Blinn-Phong): a bright highlight where the surface is
  // angled to bounce the light straight at the camera. Rather than
  // computing the true reflection vector, Blinn-Phong compares the
  // normal to the "halfway vector" between light and view directions
  // -- cheaper, and close enough that it's the standard approximation.
  // Raising to a high power (shininess) squeezes the bright region
  // down to a tight highlight instead of a broad glow.
  let halfVec = normalize(lightDir + viewDir);
  let shininess = 100.0;
  let specular = pow(max(dot(normal, halfVec), 0.0), shininess);

  // Rim light: brightens edges that face *away* from the camera --
  // the opposite condition from specular. This fakes the way real
  // translucent wax glows brightest at its silhouette, backlit by
  // light scattering through it, and is a cheap trick for making
  // rounded shapes read as soft/glowing rather than hard plastic.
  // let rimAmount = 0.0;
  let rimAmount = pow(1.0 - max(dot(normal, viewDir), 0.0), 2.0);

  let ambient = 0.1;
  let baseColor = vec3f(1.0, 0.35, 0.2);
  let lightColor = vec3f(1.0, 0.95, 0.85);
  let rimColor = vec3f(1.0, 0.5, 0.3);

  var color = baseColor * (ambient + diffuse * 0.9);
  color += lightColor * specular * 0.6;
  color += rimColor * rimAmount * 0.4;

  return vec4f(color, 1.0);
}
