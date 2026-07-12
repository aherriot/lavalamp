// Substituted by JS before shader creation (device.createShaderModule),
// since WGSL array sizes and @workgroup_size must be known at shader
// compile time -- a GUI-driven "blob count" slider can't just be a
// uniform like everything else here. Changing it means rebuilding the
// shader module (and every pipeline/bind group that references it)
// from scratch with the new count baked in.
const PHYSICS_BLOB_COUNT = __BLOB_COUNT__u;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

struct SimParams {
  time: f32,
  dt: f32,
  resolution: vec2f,
  cameraAzimuth: f32,
  cameraElevation: f32,
  maxSteps: f32,
  hitEpsilon: f32,
  // xyz = light position, w = bloom brightness threshold. Reusing the
  // otherwise-unused .w channel of each vec4 below instead of growing
  // the struct further with more scalars (which would need careful
  // re-padding) -- an intentional, slightly unconventional packing
  // choice, called out here so it doesn't look like a mistake.
  lightPos: vec4f,
  coolColor: vec4f, // rgb = lava gradient cool stop, w = bloom intensity
  warmColor: vec4f, // rgb = lava gradient warm stop, w unused
  hotColor: vec4f, // rgb = lava gradient hot stop, w unused
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

  // The spring's target height sits at the bottom by default and only
  // rises once heat climbs above riseThreshold -- a real lava lamp's
  // wax needs to sit near the bulb and warm up for a while before it's
  // buoyant enough to rise, then cools and sinks back once away from
  // the heat. Since heat is noise centered around 0, it spends most of
  // its time below riseThreshold, so targetY spends most of its time
  // at the bottom too, with shorter, less frequent excursions to the
  // top -- asymmetric, unlike the old symmetric +-heat mapping which
  // gave top and bottom equal billing. smoothstep (rather than a hard
  // cutoff) keeps the transition itself smooth once heat does cross
  // the threshold. The upper bound is deliberately much lower than
  // heat's theoretical max of 1.0: this value noise rarely swings all
  // the way to its extremes (this value noise clusters toward the
  // middle), so a wide 0.35..1.0 range meant targetT almost
  // never fully saturated and blobs only ever rose partway. Narrowing
  // it to 0.35..0.55 means a realistically-achievable heat excursion
  // is enough to send the target all the way to the top.
  let restoreStrength = 0.22;
  let riseThreshold = 0.25;
  let targetT = smoothstep(riseThreshold, 0.55, heat);
  // Bottom target is much lower than the top one (asymmetric on
  // purpose) so resting blobs settle down near the base of the glass,
  // not stop halfway down the container.
  let targetY = mix(-1.8, 1.5, targetT);
  var ay = (targetY - b.pos.y) * restoreStrength;
  var ax = 0.0;

  // Soft wall repulsion: an actual force pushing blobs away from the
  // container walls, growing smoothly (quadratically) the closer they
  // get, rather than relying only on the hard clamp-and-bounce at the
  // very end of this function. That clamp only ever *corrects* a
  // position after the fact, once a blob has already reached the
  // boundary; this instead discourages it from getting that close in
  // the first place, so walls are actually felt as resistance during
  // the approach.
  let boundX = 0.65;
  // Raised from 1.35 so the lower target (-1.8 above) has room to
  // actually be reached instead of getting clamped well short of it;
  // still safely inside the glass capsule's rounded-cap tip at ~2.25.
  let boundY = 2.0;
  let wallMarginX = boundX * 0.7;
  let wallMarginY = boundY * 0.7;
  let wallStrength = 8.0;
  if (abs(b.pos.x) > wallMarginX) {
    let excess = abs(b.pos.x) - wallMarginX;
    ax -= sign(b.pos.x) * excess * excess * wallStrength;
  }
  if (abs(b.pos.y) > wallMarginY) {
    let excess = abs(b.pos.y) - wallMarginY;
    ay -= sign(b.pos.y) * excess * excess * wallStrength;
  }

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

  // Hard clamp as a last-resort safety net -- the soft wall repulsion
  // above should normally keep blobs from ever reaching this, but
  // velocity can still carry one past it in a single frame under
  // extreme values. boundX/boundY are declared earlier in this
  // function now, shared with the repulsion force above.
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

// ---- Render stage: 3D camera + raymarching, reading blob positions
// from the same storage buffer the compute stage above writes. ----

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

// Signed distance to a sphere: negative inside, zero on the surface,
// positive outside.
fn sdSphere(p: vec3f, radius: f32) -> f32 {
  return length(p) - radius;
}

// Smooth minimum: like min(a, b), but blends smoothly between the two
// instead of switching abruptly. smin operates on plain scalar
// distances, so it doesn't care whether those distances came from a
// 2D or 3D SDF. Blending multiple sdSphere calls with this is exactly
// what makes them merge into lava blobs
// instead of just overlapping.
fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

// Signed distance to a capsule (a cylinder with hemisphere caps): the
// lamp's glass body. `a`/`b` are the two endpoints of the cylinder's
// central axis, `r` is its radius. Declared here (ahead of sceneSDF,
// its first caller below) rather than down with the rest of the glass
// code, since sceneSDF now needs it too.
fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

fn sdGlass(p: vec3f) -> f32 {
  return sdCapsule(p, vec3f(0.0, -1.3, 0.0), vec3f(0.0, 1.3, 0.0), 0.95);
}

// Each blob's X/Y position comes from the compute shader's
// buoyancy/repulsion simulation in the storage buffer. The simulation
// itself is only 2D (it never touches a Z axis), so each blob gets a
// procedural Z offset here purely to spread them out in depth; X and
// Y are simulated, Z is not.
fn sceneSDF(p: vec3f) -> f32 {
  let k = 0.4;
  var d = 1e5;
  for (var i = 0u; i < PHYSICS_BLOB_COUNT; i++) {
    let simPos = blobsRO[i].pos;
    let fi = f32(i);

    // Spread blobs evenly across Z, centered on 0, regardless of how
    // many there are -- formula-based since PHYSICS_BLOB_COUNT can
    // change at runtime (via a GUI-triggered shader rebuild). Spacing
    // is capped so the *total* spread never exceeds ~1.3 units: at up
    // to 7 blobs this is a fixed 0.22 spacing, but beyond that it
    // shrinks automatically -- without this, a high blob count (up to
    // 24) would push outer blobs' Z far past the
    // glass's ~0.95 radius, clipping them away entirely via sceneSDF's
    // glass intersection and making them invisible.
    let zSpacing = min(0.22, 1.3 / max(f32(PHYSICS_BLOB_COUNT) - 1.0, 1.0));
    let zOffset = (fi - f32(PHYSICS_BLOB_COUNT - 1u) * 0.5) * zSpacing;
    // Base radius multiplier: 0.75x (halved from the original 1.5x),
    // then bumped up another 25% to 0.9375x. Two overlapping sine
    // terms at different frequencies/phases give more size variation
    // than a single sine and are less likely to visibly repeat across
    // many blobs than one sine alone.
    let radius = 1.2 * (0.28 + 0.14 * sin(fi * 1.7) + 0.06 * sin(fi * 4.3 + 1.0));

    let blobPos = vec3f(simPos.x, simPos.y, zOffset);
    let bd = sdSphere(p - blobPos, radius);
    d = smin(d, bd, k);
  }

  // CSG intersection with the glass interior (max of two SDFs = the
  // region inside both). Without this, a blob whose center drifts
  // near the wall pokes its far side outside the glass -- and since
  // the wax and glass are marched as two totally independent rays,
  // whichever surface happens to be hit first "wins" with a flat,
  // unnatural-looking cutoff right at the glass boundary. Intersecting
  // here instead makes the wax geometry itself bend and flatten
  // against the inside of the glass, the way real wax actually
  // deforms against a container wall.
  return max(d, sdGlass(p));
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
  for (var i = 0; i < i32(params.maxSteps); i++) {
    let p = ro + rd * t;
    let d = sceneSDF(p);
    if (d < params.hitEpsilon) {
      return t;
    }
    t += d;
    if (t > 50.0) {
      break;
    }
  }
  return -1.0; // no surface found within range: a miss
}

// Maps a blob surface's world-space height to a "cool base -> hot tip"
// lava color ramp, mimicking how real heated wax looks brighter/more
// yellow near the top of its rise and darker red lower down. Two
// linear mixes chained together (cool->warm, then warm->hot) is a
// cheap way to get a 3-stop gradient without a texture lookup.
fn lavaColor(height: f32) -> vec3f {
  let t = clamp((height + 1.0) / 2.0, 0.0, 1.0);
  if (t < 0.5) {
    return mix(params.coolColor.rgb, params.warmColor.rgb, t * 2.0);
  }
  return mix(params.warmColor.rgb, params.hotColor.rgb, (t - 0.5) * 2.0);
}

// The wax's full ambient + diffuse + specular + rim shading, as a
// standalone function so both "hit wax directly" and "hit wax seen
// through the glass" can share it.
fn shadeWax(hitPoint: vec3f, camPos: vec3f) -> vec3f {
  let normal = estimateNormal(hitPoint);

  let lightDir = normalize(params.lightPos.xyz - hitPoint);
  let viewDir = normalize(camPos - hitPoint);

  let diffuse = max(dot(normal, lightDir), 0.0);

  let halfVec = normalize(lightDir + viewDir);
  let shininess = 100.0;
  let specular = pow(max(dot(normal, halfVec), 0.0), shininess);

  let rimAmount = pow(1.0 - max(dot(normal, viewDir), 0.0), 2.0);

  let ambient = 0.1;
  let baseColor = lavaColor(hitPoint.y);
  let lightColor = vec3f(1.0, 0.95, 0.85);
  let rimColor = vec3f(1.0, 0.5, 0.3);

  var color = baseColor * (ambient + diffuse * 0.9);
  color += lightColor * specular * 0.6;
  color += rimColor * rimAmount * 0.4;
  return color;
}

// A simple vertical sky gradient instead of flat black, so there's
// still something to look at, and something for the glass to reflect.
fn skyColor(rd: vec3f) -> vec3f {
  let skyT = rd.y * 0.5 + 0.5;
  return mix(vec3f(0.05, 0.05, 0.1), vec3f(0.25, 0.28, 0.4), skyT);
}

fn estimateGlassNormal(p: vec3f) -> vec3f {
  let e = 0.001;
  return normalize(vec3f(
    sdGlass(p + vec3f(e, 0.0, 0.0)) - sdGlass(p - vec3f(e, 0.0, 0.0)),
    sdGlass(p + vec3f(0.0, e, 0.0)) - sdGlass(p - vec3f(0.0, e, 0.0)),
    sdGlass(p + vec3f(0.0, 0.0, e)) - sdGlass(p - vec3f(0.0, 0.0, e)),
  ));
}

// A second, separate sphere-tracing loop against sdGlass instead of
// sceneSDF. WGSL has no function pointers, so rather than making
// raymarch generic, the glass gets its own small copy -- a bit of
// duplication in exchange for staying simple to read.
fn raymarchGlass(ro: vec3f, rd: vec3f) -> f32 {
  var t = 0.0;
  for (var i = 0; i < i32(params.maxSteps); i++) {
    let p = ro + rd * t;
    let d = sdGlass(p);
    if (d < params.hitEpsilon) {
      return t;
    }
    t += d;
    if (t > 50.0) {
      break;
    }
  }
  return -1.0;
}

// Signed distance to a capped cone (truncated cone / frustum) aligned
// on the Y axis, spanning local y in [-h, h]: radius r1 at the bottom
// (y=-h), r2 at the top (y=+h). Standard formula (Inigo Quilez's
// distance-function reference); WGSL has no C-style ternary, so the
// two conditional picks below use select(falseValue, trueValue, cond).
fn sdCappedCone(p: vec3f, h: f32, r1: f32, r2: f32) -> f32 {
  let q = vec2f(length(p.xz), p.y);
  let k1 = vec2f(r2, h);
  let k2 = vec2f(r2 - r1, 2.0 * h);
  let caX = q.x - min(q.x, select(r2, r1, q.y < 0.0));
  let ca = vec2f(caX, abs(q.y) - h);
  let t = clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
  let cb = q - k1 + k2 * t;
  let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
  return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

// The classic lava lamp base shape: two frustums joined at a narrow
// waist -- wide where it meets the glass, pinching in, then flaring
// back out to a wide flat foot. Built as a union (min) of two
// sdCappedCone calls that share the same radius at the waist, so they
// meet with no visible seam.
fn sdBase(p: vec3f) -> f32 {
  let baseCenter = vec3f(0.0, -2.7, 0.0);
  let p2 = p - baseCenter;

  let waistRadius = 0.52;
  let topRadius = 0.95;
  let footRadius = 1.05;
  let upperHalf = 0.6;
  let lowerHalf = 0.6;

  // Upper frustum: narrow at the waist (bottom), flares out to meet
  // the glass at the top.
  let dUpper = sdCappedCone(
    p2 - vec3f(0.0, upperHalf, 0.0),
    upperHalf,
    waistRadius,
    topRadius,
  );

  // Lower frustum: mirrored -- narrow at the waist (top), flares out
  // to a wide flat foot at the bottom.
  let dLower = sdCappedCone(
    p2 - vec3f(0.0, -lowerHalf, 0.0),
    lowerHalf,
    footRadius,
    waistRadius,
  );

  return min(dUpper, dLower);
}

fn estimateBaseNormal(p: vec3f) -> vec3f {
  let e = 0.001;
  return normalize(vec3f(
    sdBase(p + vec3f(e, 0.0, 0.0)) - sdBase(p - vec3f(e, 0.0, 0.0)),
    sdBase(p + vec3f(0.0, e, 0.0)) - sdBase(p - vec3f(0.0, e, 0.0)),
    sdBase(p + vec3f(0.0, 0.0, e)) - sdBase(p - vec3f(0.0, 0.0, e)),
  ));
}

fn raymarchBase(ro: vec3f, rd: vec3f) -> f32 {
  var t = 0.0;
  for (var i = 0; i < i32(params.maxSteps); i++) {
    let p = ro + rd * t;
    let d = sdBase(p);
    if (d < params.hitEpsilon) {
      return t;
    }
    t += d;
    if (t > 50.0) {
      break;
    }
  }
  return -1.0;
}

// Unlike the glass, the base is a plain opaque solid: no fresnel, no
// refraction, just ambient + diffuse + specular -- the same shape of
// shading as shadeWax, minus the rim light and lava color ramp.
fn shadeBase(hitPoint: vec3f, camPos: vec3f) -> vec3f {
  let normal = estimateBaseNormal(hitPoint);
  let lightDir = normalize(params.lightPos.xyz - hitPoint);
  let viewDir = normalize(camPos - hitPoint);

  let diffuse = max(dot(normal, lightDir), 0.0);
  let halfVec = normalize(lightDir + viewDir);
  let ndoth = max(dot(normal, halfVec), 0.0);
  // Two specular lobes layered together, the way a well-polished
  // metal highlight actually looks: a broad, soft sheen (low exponent)
  // giving the surface some general glow near the light, plus a much
  // tighter, sharper core (high exponent) right where the reflection
  // points exactly at the camera -- a single lobe alone is either
  // "soft plastic" (low exponent) or "a barely-visible pinpoint" (high
  // exponent with no broad companion); together they read as a crisp
  // mirror-like glint sitting inside a gentler glow.
  let specularBroad = pow(ndoth, 140.0);
  let specularTight = pow(ndoth, 700.0);

  let ambient = 0.18;
  // Neutral metallic grey (brushed aluminum/chrome) instead of the
  // previous warm dark bronze, plus a brighter ambient term so it
  // doesn't read as near-black in shadow.
  let metalColor = vec3f(0.55, 0.56, 0.58);
  var color = metalColor * (ambient + diffuse * 0.8);
  color += vec3f(1.0, 1.0, 1.0) * specularBroad * 1.0;
  color += vec3f(1.0, 1.0, 1.0) * specularTight * 2.5;

  // Metallic fresnel rim: real metals reflect more strongly at
  // grazing angles, same geometric idea as the glass's Fresnel term,
  // but tinted by the metal's own color rather than staying neutral
  // -- that color tint is exactly what visually
  // distinguishes "shiny metal" from "shiny plastic/glass". This adds
  // reflectivity across the whole silhouette edge, not just a single
  // point highlight, which sells "shiny" far more than the specular
  // glint alone.
  let fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);
  color += metalColor * fresnel * 0.9;

  return color;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let aspect = params.resolution.x / params.resolution.y;

  // Centered, aspect-corrected screen coordinate in roughly -1..1,
  // feeding a 3D camera instead of an SDF directly.
  var screen = in.uv * 2.0 - 1.0;
  screen.x *= aspect;

  // Orbit camera, driven by JS-tracked azimuth/elevation (mouse-drag
  // control, auto-rotating slowly while idle) rather than time alone.
  // Standard spherical-to-Cartesian conversion: azimuth sweeps around
  // the vertical axis, elevation tilts up/down toward the poles.
  let camDist = 6.0;
  let camPos = vec3f(
    camDist * cos(params.cameraElevation) * sin(params.cameraAzimuth),
    camDist * sin(params.cameraElevation),
    camDist * cos(params.cameraElevation) * cos(params.cameraAzimuth),
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

  // The glass fully encloses the wax, so a ray that reaches any wax
  // always reaches the glass shell first -- meaning tGlass, if it
  // hits, is always the correct "first surface" for this pixel.
  let tGlass = raymarchGlass(camPos, rayDir);

  // The base is a separate, unrelated solid -- not inside the glass --
  // so whichever of tBase/tGlass is closer determines what's actually
  // visible first along this ray.
  let tBase = raymarchBase(camPos, rayDir);
  if (tBase >= 0.0 && (tGlass < 0.0 || tBase < tGlass)) {
    return vec4f(shadeBase(camPos + rayDir * tBase, camPos), 1.0);
  }

  if (tGlass < 0.0) {
    return vec4f(skyColor(rayDir), 1.0);
  }

  // "Fake" refraction: rather than actually bending the ray at the
  // glass surface (which would mean marching a second ray from inside
  // the glass, at real cost), just reuse the *undisturbed* ray's wax
  // hit. It's not physically correct refraction, but for a thin-ish
  // glass shell the visual difference is minor, and it's essentially
  // free since raymarch() was going to run anyway.
  let tBlob = raymarch(camPos, rayDir);
  var innerColor: vec3f;
  if (tBlob >= 0.0) {
    innerColor = shadeWax(camPos + rayDir * tBlob, camPos);
  } else {
    // No wax along this ray: a warm, dim liquid color instead of a
    // hard edge where the glass would otherwise show empty space.
    innerColor = vec3f(0.22, 0.07, 0.05);
  }

  let glassPoint = camPos + rayDir * tGlass;
  let glassNormal = estimateGlassNormal(glassPoint);
  let viewDir = normalize(camPos - glassPoint);

  // Fresnel (Schlick's approximation): real glass reflects more and
  // transmits less the more glancing the viewing angle is -- almost a
  // mirror at grazing angles. F0 is the reflectance straight-on
  // (head-on incidence): real glass reflects roughly 4% of light even
  // when viewed dead-on, which is what made the capsule read as a
  // flat, opaque color before -- with no baseline reflectivity, the
  // "glass" contributed nothing when looking straight through it.
  let cosTheta = max(dot(glassNormal, viewDir), 0.0);
  let F0 = 0.04;
  let fresnel = F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);

  let reflectDir = reflect(-viewDir, glassNormal);
  let reflectionColor = skyColor(reflectDir);

  let glassTint = vec3f(0.85, 0.95, 0.9);
  let refractedColor = innerColor * glassTint;

  var finalColor = mix(refractedColor, reflectionColor, fresnel);

  // A direct specular glint from the light source hitting the glass
  // surface itself -- the sky reflection alone never picks this up,
  // since the light is a point light, not part of the sky. This bright,
  // tight highlight is often the single strongest cue that a surface
  // is "glassy/wet" rather than a flat colored material.
  let glassLightDir = normalize(params.lightPos.xyz - glassPoint);
  let glassHalfVec = normalize(glassLightDir + viewDir);
  let glassSpecular = pow(max(dot(glassNormal, glassHalfVec), 0.0), 200.0);
  finalColor += vec3f(1.0) * glassSpecular;
  return vec4f(finalColor, 1.0);
}

// ---- Bloom post-processing: two extra fullscreen passes reusing the
// same vs_main fullscreen-triangle trick. fs_main above rendered the
// actual scene into an offscreen texture instead of the canvas; these
// two passes read that texture back as input. ----

@group(0) @binding(2) var texSampler: sampler;
@group(0) @binding(3) var sceneTex: texture_2d<f32>;

// Extracts only the bright parts of the scene (the glass specular
// glint, wax highlights) and blurs them. Real bloom pipelines usually
// separate "extract" and "blur" into their own passes (often several,
// at shrinking resolutions) for a smoother glow; this folds both into
// one pass with a single small blur kernel, trading some quality for
// simplicity.
@fragment
fn bloomExtract_fs(in: VertexOutput) -> @location(0) vec4f {
  let texel = 1.0 / params.resolution;
  let radius = 3.0;

  // vs_main's `uv` is y-up (uv.y = 1 at the top of the screen), but
  // textureSample's v-coordinate is y-down (v = 0 is the top texel
  // row). Rasterizing straight to the canvas (fs_main) never notices
  // this mismatch since nothing samples a texture there -- but any
  // pass that samples a texture written by a previous pass needs the
  // v flipped to actually land on the intended pixel.
  let sampleUV = vec2f(in.uv.x, 1.0 - in.uv.y);

  var offsets = array<vec2f, 9>(
    vec2f(-1.0, -1.0), vec2f(0.0, -1.0), vec2f(1.0, -1.0),
    vec2f(-1.0, 0.0), vec2f(0.0, 0.0), vec2f(1.0, 0.0),
    vec2f(-1.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 1.0),
  );

  var sum = vec3f(0.0);
  for (var i = 0; i < 9; i++) {
    let uv = sampleUV + offsets[i] * texel * radius;
    let c = textureSample(sceneTex, texSampler, uv).rgb;

    // Threshold: only pixels already close to full brightness
    // contribute to the glow, otherwise the whole image would bloom.
    // Reused from params.lightPos.w -- see the SimParams struct note.
    let brightness = max(c.r, max(c.g, c.b));
    let excess = max(brightness - params.lightPos.w, 0.0);
    sum += c * (excess / max(brightness, 0.0001));
  }

  return vec4f(sum / 9.0, 1.0);
}

@group(0) @binding(4) var bloomTex: texture_2d<f32>;

// Adds the blurred bright-pass on top of the original scene. This
// additive combination is what makes bright areas visibly "glow" --
// spilling light into the darker pixels around them -- rather than
// just being clipped at pure white.
@fragment
fn composite_fs(in: VertexOutput) -> @location(0) vec4f {
  // Same y-up-vs-y-down mismatch as bloomExtract_fs, applied to both
  // texture reads.
  let sampleUV = vec2f(in.uv.x, 1.0 - in.uv.y);
  let scene = textureSample(sceneTex, texSampler, sampleUV).rgb;
  let bloom = textureSample(bloomTex, texSampler, sampleUV).rgb;
  // Intensity reused from params.coolColor.w -- see SimParams struct note.
  return vec4f(scene + bloom * params.coolColor.w, 1.0);
}
