/**
 * OrbitControls for three.js r158 (bundled locally for offline use).
 * Original: https://threejs.org
 */
(function () {
  class OrbitControls extends THREE.EventDispatcher {
    constructor(object, domElement) {
      super();
      this.object = object;
      this.domElement = domElement;
      this.enabled = true;
      this.target = new THREE.Vector3();
      this.minDistance = 0;
      this.maxDistance = Infinity;
      this.minZoom = 0;
      this.maxZoom = Infinity;
      this.minPolarAngle = 0;
      this.maxPolarAngle = Math.PI;
      this.minAzimuthAngle = -Infinity;
      this.maxAzimuthAngle = Infinity;
      this.enableDamping = true;
      this.dampingFactor = 0.08;
      this.enableZoom = true;
      this.zoomSpeed = 1.0;
      this.enableRotate = true;
      this.rotateSpeed = 0.9;
      this.enablePan = true;
      this.panSpeed = 0.7;
      this.screenSpacePanning = true;
      this.autoRotate = false;
      this.autoRotateSpeed = 2.0;
      this.keys = { LEFT: "ArrowLeft", UP: "ArrowUp", RIGHT: "ArrowRight", BOTTOM: "ArrowDown" };
      this.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
      this.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };

      const scope = this;
      const STATE = { NONE: -1, ROTATE: 0, DOLLY: 1, PAN: 2, TOUCH_ROTATE: 3, TOUCH_PAN: 4, TOUCH_DOLLY_PAN: 5 };
      let state = STATE.NONE;

      const EPS = 1e-6;
      const spherical = new THREE.Spherical();
      const sphericalDelta = new THREE.Spherical();
      let scale = 1;
      const panOffset = new THREE.Vector3();
      let zoomChanged = false;
      const rotateStart = new THREE.Vector2();
      const rotateEnd = new THREE.Vector2();
      const rotateDelta = new THREE.Vector2();
      const panStart = new THREE.Vector2();
      const panEnd = new THREE.Vector2();
      const panDelta = new THREE.Vector2();
      const dollyStart = new THREE.Vector2();
      const dollyEnd = new THREE.Vector2();
      const dollyDelta = new THREE.Vector2();

      this.update = (function () {
        const offset = new THREE.Vector3();
        const quat = new THREE.Quaternion().setFromUnitVectors(object.up, new THREE.Vector3(0, 1, 0));
        const quatInverse = quat.clone().invert();
        const lastPosition = new THREE.Vector3();
        const lastQuaternion = new THREE.Quaternion();
        return function update() {
          const position = scope.object.position;
          offset.copy(position).sub(scope.target);
          offset.applyQuaternion(quat);
          spherical.setFromVector3(offset);
          if (scope.autoRotate && state === STATE.NONE) rotateLeft(getAutoRotationAngle());
          // With damping the delta is consumed a fraction at a time and decayed
          // below, so only that fraction may be applied per frame. Applying the
          // whole delta *and* decaying it (as this copy previously did) makes
          // each drag keep travelling long after it ends — the model overshoots
          // and feels uncontrollable, worst on a small phone canvas.
          if (scope.enableDamping) {
            spherical.theta += sphericalDelta.theta * scope.dampingFactor;
            spherical.phi += sphericalDelta.phi * scope.dampingFactor;
          } else {
            spherical.theta += sphericalDelta.theta;
            spherical.phi += sphericalDelta.phi;
          }
          spherical.theta = Math.max(scope.minAzimuthAngle, Math.min(scope.maxAzimuthAngle, spherical.theta));
          spherical.phi = Math.max(scope.minPolarAngle, Math.min(scope.maxPolarAngle, spherical.phi));
          spherical.makeSafe();
          spherical.radius *= scale;
          spherical.radius = Math.max(scope.minDistance, Math.min(scope.maxDistance, spherical.radius));
          // Same reasoning as the rotation above.
          if (scope.enableDamping) scope.target.addScaledVector(panOffset, scope.dampingFactor);
          else scope.target.add(panOffset);
          offset.setFromSpherical(spherical);
          offset.applyQuaternion(quatInverse);
          position.copy(scope.target).add(offset);
          scope.object.lookAt(scope.target);
          if (scope.enableDamping) {
            sphericalDelta.theta *= 1 - scope.dampingFactor;
            sphericalDelta.phi *= 1 - scope.dampingFactor;
            panOffset.multiplyScalar(1 - scope.dampingFactor);
          } else {
            sphericalDelta.set(0, 0, 0);
            panOffset.set(0, 0, 0);
          }
          scale = 1;
          if (zoomChanged ||
            lastPosition.distanceToSquared(scope.object.position) > EPS ||
            8 * (1 - lastQuaternion.dot(scope.object.quaternion)) > EPS) {
            scope.dispatchEvent({ type: "change" });
            lastPosition.copy(scope.object.position);
            lastQuaternion.copy(scope.object.quaternion);
            zoomChanged = false;
            return true;
          }
          return false;
        };
      })();

      this.dispose = function () {
        scope.domElement.removeEventListener("contextmenu", onContextMenu);
        scope.domElement.removeEventListener("pointerdown", onPointerDown);
        scope.domElement.removeEventListener("pointercancel", onPointerCancel);
        scope.domElement.removeEventListener("wheel", onMouseWheel);
        scope.domElement.removeEventListener("pointermove", onPointerMove);
        scope.domElement.removeEventListener("pointerup", onPointerUp);
        window.removeEventListener("keydown", onKeyDown);
      };

      function getAutoRotationAngle() {
        return ((2 * Math.PI) / 60 / 60) * scope.autoRotateSpeed;
      }
      function getZoomScale() {
        return Math.pow(0.95, scope.zoomSpeed);
      }
      function rotateLeft(angle) {
        sphericalDelta.theta -= angle;
      }
      function rotateUp(angle) {
        sphericalDelta.phi -= angle;
      }
      const panLeft = (function () {
        const v = new THREE.Vector3();
        return function panLeft(distance, objectMatrix) {
          v.setFromMatrixColumn(objectMatrix, 0);
          v.multiplyScalar(-distance);
          panOffset.add(v);
        };
      })();
      const panUp = (function () {
        const v = new THREE.Vector3();
        return function panUp(distance, objectMatrix) {
          if (scope.screenSpacePanning) {
            v.setFromMatrixColumn(objectMatrix, 1);
          } else {
            v.setFromMatrixColumn(objectMatrix, 0);
            v.crossVectors(scope.object.up, v);
          }
          v.multiplyScalar(distance);
          panOffset.add(v);
        };
      })();
      const pan = (function () {
        const offset = new THREE.Vector3();
        return function pan(deltaX, deltaY) {
          const element = scope.domElement;
          if (scope.object.isPerspectiveCamera) {
            const position = scope.object.position;
            offset.copy(position).sub(scope.target);
            let targetDistance = offset.length();
            targetDistance *= Math.tan(((scope.object.fov / 2) * Math.PI) / 180.0);
            panLeft((2 * deltaX * targetDistance) / element.clientHeight, scope.object.matrix);
            panUp((2 * deltaY * targetDistance) / element.clientHeight, scope.object.matrix);
          } else if (scope.object.isOrthographicCamera) {
            panLeft((deltaX * (scope.object.right - scope.object.left)) / scope.object.zoom / element.clientWidth, scope.object.matrix);
            panUp((deltaY * (scope.object.top - scope.object.bottom)) / scope.object.zoom / element.clientHeight, scope.object.matrix);
          } else {
            console.warn("OrbitControls: Unsupported camera type");
            scope.enablePan = false;
          }
        };
      })();

      function dollyOut(dollyScale) {
        if (scope.object.isPerspectiveCamera) scale *= dollyScale;
        else if (scope.object.isOrthographicCamera) {
          scope.object.zoom = Math.max(scope.minZoom, Math.min(scope.maxZoom, scope.object.zoom * dollyScale));
          scope.object.updateProjectionMatrix();
          zoomChanged = true;
        }
      }
      function dollyIn(dollyScale) {
        if (scope.object.isPerspectiveCamera) scale /= dollyScale;
        else if (scope.object.isOrthographicCamera) {
          scope.object.zoom = Math.max(scope.minZoom, Math.min(scope.maxZoom, scope.object.zoom / dollyScale));
          scope.object.updateProjectionMatrix();
          zoomChanged = true;
        }
      }
      function handleMouseDownRotate(event) {
        rotateStart.set(event.clientX, event.clientY);
      }
      function handleMouseDownDolly(event) {
        dollyStart.set(event.clientX, event.clientY);
      }
      function handleMouseDownPan(event) {
        panStart.set(event.clientX, event.clientY);
      }
      function handleMouseMoveRotate(event) {
        rotateEnd.set(event.clientX, event.clientY);
        rotateDelta.subVectors(rotateEnd, rotateStart).multiplyScalar(scope.rotateSpeed / scope.domElement.clientHeight);
        rotateLeft((2 * Math.PI * rotateDelta.x));
        rotateUp((2 * Math.PI * rotateDelta.y));
        rotateStart.copy(rotateEnd);
      }
      function handleMouseMoveDolly(event) {
        dollyEnd.set(event.clientX, event.clientY);
        dollyDelta.subVectors(dollyEnd, dollyStart);
        if (dollyDelta.y > 0) dollyOut(getZoomScale());
        else if (dollyDelta.y < 0) dollyIn(getZoomScale());
        dollyStart.copy(dollyEnd);
      }
      function handleMouseMovePan(event) {
        panEnd.set(event.clientX, event.clientY);
        panDelta.subVectors(panEnd, panStart).multiplyScalar(scope.panSpeed);
        pan(panDelta.x, panDelta.y);
        panStart.copy(panEnd);
      }
      function handleMouseWheel(event) {
        if (event.deltaY < 0) dollyIn(getZoomScale());
        else if (event.deltaY > 0) dollyOut(getZoomScale());
      }
      function handleKeyDown(event) {
        let needsUpdate = false;
        switch (event.code) {
          case scope.keys.UP:
            pan(0, scope.keyPanSpeed);
            needsUpdate = true;
            break;
          case scope.keys.BOTTOM:
            pan(0, -scope.keyPanSpeed);
            needsUpdate = true;
            break;
          case scope.keys.LEFT:
            pan(scope.keyPanSpeed, 0);
            needsUpdate = true;
            break;
          case scope.keys.RIGHT:
            pan(-scope.keyPanSpeed, 0);
            needsUpdate = true;
            break;
        }
        if (needsUpdate) {
          event.preventDefault();
          scope.update();
        }
      }
      function handleTouchStartRotate(event) {
        rotateStart.set(event.touches[0].pageX, event.touches[0].pageY);
      }
      function handleTouchStartPan(event) {
        panStart.set(event.touches[0].pageX, event.touches[0].pageY);
      }
      function handleTouchStartDollyPan(event) {
        if (scope.enableZoom === false) return;
        const dx = event.touches[0].pageX - event.touches[1].pageX;
        const dy = event.touches[0].pageY - event.touches[1].pageY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        dollyStart.set(0, distance);
        const x = 0.5 * (event.touches[0].pageX + event.touches[1].pageX);
        const y = 0.5 * (event.touches[0].pageY + event.touches[1].pageY);
        panStart.set(x, y);
      }
      function handleTouchMoveRotate(event) {
        rotateEnd.set(event.touches[0].pageX, event.touches[0].pageY);
        rotateDelta.subVectors(rotateEnd, rotateStart).multiplyScalar(scope.rotateSpeed / scope.domElement.clientHeight);
        rotateLeft((2 * Math.PI * rotateDelta.x));
        rotateUp((2 * Math.PI * rotateDelta.y));
        rotateStart.copy(rotateEnd);
      }
      function handleTouchMovePan(event) {
        panEnd.set(event.touches[0].pageX, event.touches[0].pageY);
        panDelta.subVectors(panEnd, panStart).multiplyScalar(scope.panSpeed);
        pan(panDelta.x, panDelta.y);
        panStart.copy(panEnd);
      }
      function handleTouchMoveDollyPan(event) {
        const dx = event.touches[0].pageX - event.touches[1].pageX;
        const dy = event.touches[0].pageY - event.touches[1].pageY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        dollyEnd.set(0, distance);
        dollyDelta.subVectors(dollyEnd, dollyStart);
        if (dollyDelta.y > 0) dollyOut(getZoomScale());
        else if (dollyDelta.y < 0) dollyIn(getZoomScale());
        dollyStart.copy(dollyEnd);
        const x = 0.5 * (event.touches[0].pageX + event.touches[1].pageX);
        const y = 0.5 * (event.touches[0].pageY + event.touches[1].pageY);
        panEnd.set(x, y);
        panDelta.subVectors(panEnd, panStart).multiplyScalar(scope.panSpeed);
        pan(panDelta.x, panDelta.y);
        panStart.copy(panEnd);
      }

      function onPointerDown(event) {
        if (!scope.enabled) return;
        // Only mouse/pen are driven through pointer events; touch is handled by
        // the touch listeners below. Calling preventDefault or capturing the
        // pointer for a touch would swallow the browser's own scroll gesture,
        // which strands the user on a page where the viewer fills the screen.
        switch (event.pointerType) {
          case "mouse":
          case "pen":
            event.preventDefault();
            scope.domElement.setPointerCapture(event.pointerId);
            onMouseDown(event);
            break;
        }
      }
      function onMouseDown(event) {
        let mouseAction;
        if (event.button === 0) mouseAction = scope.mouseButtons.LEFT;
        else if (event.button === 1) mouseAction = scope.mouseButtons.MIDDLE;
        else if (event.button === 2) mouseAction = scope.mouseButtons.RIGHT;
        switch (mouseAction) {
          case THREE.MOUSE.DOLLY:
            if (!scope.enableZoom) return;
            handleMouseDownDolly(event);
            state = STATE.DOLLY;
            break;
          case THREE.MOUSE.ROTATE:
            if (!scope.enableRotate) return;
            handleMouseDownRotate(event);
            state = STATE.ROTATE;
            break;
          case THREE.MOUSE.PAN:
            if (!scope.enablePan) return;
            handleMouseDownPan(event);
            state = STATE.PAN;
            break;
          default:
            state = STATE.NONE;
        }
      }
      function onPointerMove(event) {
        if (!scope.enabled) return;
        if (state === STATE.NONE) return;
        switch (state) {
          case STATE.ROTATE:
            if (!scope.enableRotate) return;
            handleMouseMoveRotate(event);
            break;
          case STATE.DOLLY:
            if (!scope.enableZoom) return;
            handleMouseMoveDolly(event);
            break;
          case STATE.PAN:
            if (!scope.enablePan) return;
            handleMouseMovePan(event);
            break;
        }
      }
      function onPointerUp(event) {
        // Only mouse/pen are ever captured, and releasing an uncaptured
        // pointer throws.
        if (event.pointerType === "mouse" || event.pointerType === "pen") {
          try { scope.domElement.releasePointerCapture(event.pointerId); } catch (e) {}
        }
        state = STATE.NONE;
      }
      function onPointerCancel() {
        state = STATE.NONE;
      }
      function onMouseWheel(event) {
        if (!scope.enabled || !scope.enableZoom || state !== STATE.NONE) return;
        event.preventDefault();
        handleMouseWheel(event);
        scope.update();
      }
      function onKeyDown(event) {
        if (!scope.enabled || !scope.enablePan) return;
        handleKeyDown(event);
      }
      function onContextMenu(event) {
        event.preventDefault();
      }

      // Touch support. The handleTouch* helpers above already existed but
      // nothing called them, so the viewer was inert on phones. These
      // listeners are deliberately passive: they never call preventDefault,
      // so the browser's own gesture handling always wins. Combined with
      // `touch-action: pan-y` on the canvas that means a vertical swipe
      // scrolls the page while a sideways drag rotates the model, and two
      // fingers pinch to zoom.
      function onTouchStart(event) {
        if (!scope.enabled) return;
        switch (event.touches.length) {
          case 1:
            if (scope.enableRotate === false) { state = STATE.NONE; return; }
            handleTouchStartRotate(event);
            state = STATE.TOUCH_ROTATE;
            break;
          case 2:
            if (scope.enableZoom === false && scope.enablePan === false) { state = STATE.NONE; return; }
            handleTouchStartDollyPan(event);
            state = STATE.TOUCH_DOLLY_PAN;
            break;
          default:
            state = STATE.NONE;
        }
      }
      function onTouchMove(event) {
        if (!scope.enabled) return;
        switch (state) {
          case STATE.TOUCH_ROTATE:
            if (scope.enableRotate === false || event.touches.length !== 1) return;
            handleTouchMoveRotate(event);
            scope.update();
            break;
          case STATE.TOUCH_DOLLY_PAN:
            if (event.touches.length !== 2) return;
            handleTouchMoveDollyPan(event);
            scope.update();
            break;
          default:
            state = STATE.NONE;
        }
      }
      function onTouchEnd() {
        state = STATE.NONE;
      }

      scope.domElement.addEventListener("contextmenu", onContextMenu);
      scope.domElement.addEventListener("pointerdown", onPointerDown);
      scope.domElement.addEventListener("pointercancel", onPointerCancel);
      scope.domElement.addEventListener("wheel", onMouseWheel, { passive: false });
      scope.domElement.addEventListener("pointermove", onPointerMove);
      scope.domElement.addEventListener("pointerup", onPointerUp);
      scope.domElement.addEventListener("touchstart", onTouchStart, { passive: true });
      scope.domElement.addEventListener("touchmove", onTouchMove, { passive: true });
      scope.domElement.addEventListener("touchend", onTouchEnd, { passive: true });
      scope.domElement.addEventListener("touchcancel", onTouchEnd, { passive: true });
      window.addEventListener("keydown", onKeyDown);

      this.update();
    }
  }

  THREE.OrbitControls = OrbitControls;
})();
