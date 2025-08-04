import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.157/build/three.module.js";

let scene, camera, renderer, prosthetic;

init();
animate();
connectWebSocket();

function init() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  camera.position.z = 2;

  renderer = new THREE.WebGLRenderer();
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  const light = new THREE.DirectionalLight(0xffffff, 1);
  light.position.set(0, 0, 2).normalize();
  scene.add(light);

  // 仮の義足：円柱
  const geometry = new THREE.CylinderGeometry(0.02, 0.02, 0.3, 32);
  const material = new THREE.MeshStandardMaterial({ color: 0x8888ff });
  prosthetic = new THREE.Mesh(geometry, material);
  scene.add(prosthetic);
}

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

function connectWebSocket() {
  const socket = new WebSocket("ws://localhost:8765");
  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const knee = data.knee;
    const ankle = data.ankle;

    // 中心位置
    prosthetic.position.set(
      (knee.x + ankle.x) / 2,
      -(knee.y + ankle.y) / 2, // y軸反転
      -(knee.z + ankle.z) / 2 // z軸反転
    );

    // ベクトル方向に合わせて回転
    const dx = ankle.x - knee.x;
    const dy = ankle.y - knee.y;
    const dz = ankle.z - knee.z;
    const direction = new THREE.Vector3(dx, -dy, -dz).normalize();

    const up = new THREE.Vector3(0, 1, 0);
    const quaternion = new THREE.Quaternion().setFromUnitVectors(up, direction);
    prosthetic.setRotationFromQuaternion(quaternion);

    // 長さ更新（オプション）
    const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
    prosthetic.scale.set(1, length / 0.3, 1); // 0.3 = 初期長さ
  };
}
