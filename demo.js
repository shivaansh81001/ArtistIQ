const canvas = document.getElementById('draw');
    const ctx = canvas.getContext('2d');
    let drawing = false;
    let tool = 'pencil';
    let size = 8;
    let last = {x:0,y:0};

    let currentRound = 1;
    const maxRounds = 5;
    let starsEarned = 0;
    

    const stars = document.querySelectorAll('.star');
    const shapePrompts = ["circle", "triangle", "hexagon", "square", "star"];

    let idleTimer = null;
    let idleTimeLimit = 30000;

    const gameGif = document.getElementById("gameGif");
    const gifDialog = document.getElementById("gifDialog");

    function showWelcomeGif() {
        gameGif.src = "assets/welcome.gif";  
        gifDialog.textContent = "Hello, it's so nice to see you! Let's Play!";
    
        // After 3 seconds → switch to idle (default behavior)
        setTimeout(() => {
            showIdleGif();
        }, 3000);
    }
    
    function showWinGif() {
        gameGif.src = "assets/happy.gif";
        const praises = ["Yay! Good Job!", "Well Done!","Way to go!", "Fabulous!", "Amazing!", "Great Job!", "Lovely Drawing", "Woohoo! Correct!"]
        gifDialog.textContent = praises[Math.floor(Math.random() * shapePrompts.length)];
        setTimeout(() => {
          showIdleGif();
        }, 2000);
      }

    function showLoseGif(currentPrompt,predictedShape) {
        gameGif.src = "assets/nope.gif";
        let msg = `That is a good ${predictedShape.toUpperCase()} but can you draw a ${currentPrompt.toUpperCase()}`
        gifDialog.textContent = msg;
        setTimeout(() => {
          showIdleGif();
        }, 4000);
    }

    function showIdleGif() {
        gameGif.src = "assets/idle.gif";
        gifDialog.textContent = `Can you draw a ${currentPrompt}!`;
    }

    function resetIdleTimer() {
        clearTimeout(idleTimer);
        document.getElementById("idleGif").style.display = "none";

        idleTimer = setTimeout(() => {
            document.getElementById("idleGif").style.display = "block";
        }, idleTimeLimit);
    }

    function getRandomPrompt() {
        return shapePrompts[Math.floor(Math.random() * shapePrompts.length)];
    }

    function setNewPrompt() {
        const newPrompt = getRandomPrompt();
        document.getElementById("promptText").textContent = newPrompt.toUpperCase();
        return newPrompt;
    }

    function celebrate() {
        const duration = 2000;      // 2 seconds
        const end = Date.now() + duration;

        (function frame() {
            confetti({
                particleCount: 8,
                spread: 60,
                origin: { x: Math.random(), y: Math.random() * 0.2 }
            });

            if (Date.now() < end) {
                requestAnimationFrame(frame);
            }
        })();
    }
    

    function fixDpi(){
      const ratio = window.devicePixelRatio || 1;
      const w = canvas.width;
      const h = canvas.height;
      canvas.width = w * ratio;
      canvas.height = h * ratio;
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
      ctx.scale(ratio, ratio);
    }
    fixDpi();

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#111';
    ctx.lineWidth = size;

    function getPos(e){
      const rect = canvas.getBoundingClientRect();
      if (e.touches) e = e.touches[0];
      return {x: (e.clientX - rect.left), y: (e.clientY - rect.top)};
    }

    canvas.addEventListener('pointerdown', (e)=>{
      drawing = true; last = getPos(e);
    });
    window.addEventListener('pointerup', ()=>drawing=false);
    canvas.addEventListener('pointermove', (e)=>{
      if(!drawing) return;
      const p = getPos(e);
      ctx.beginPath();
      ctx.moveTo(last.x, last.y);
      ctx.lineTo(p.x, p.y);
      ctx.strokeStyle = (tool==='eraser') ? '#ffffff' : '#111';
      ctx.lineWidth = size;
      if(tool==='eraser') ctx.globalCompositeOperation = 'destination-out'; else ctx.globalCompositeOperation = 'source-over';
      ctx.stroke();
      last = p;
    });
    canvas.addEventListener("pointerdown", resetIdleTimer);
    canvas.addEventListener("pointermove", resetIdleTimer);
    canvas.addEventListener("pointerup", resetIdleTimer);
    canvas.addEventListener("pointerleave", resetIdleTimer);

    document.addEventListener("click", resetIdleTimer);

    resetIdleTimer();

    // tools
    document.getElementById('pencilBtn').addEventListener('click', ()=>{tool='pencil'; document.getElementById('pencilBtn').style.outline='4px solid rgba(0,0,0,0.12)'; document.getElementById('eraserBtn').style.outline='';});
    document.getElementById('eraserBtn').addEventListener('click', ()=>{tool='eraser'; document.getElementById('eraserBtn').style.outline='4px solid rgba(0,0,0,0.12)'; document.getElementById('pencilBtn').style.outline='';});

    document.getElementById('sizeRange').addEventListener('input', (e)=>{size = parseInt(e.target.value,10)});
    document.getElementById('clearBtn').addEventListener('click', ()=>{
      // clear to the same blue background to simulate the reference UI background
      ctx.clearRect(0,0,canvas.width,canvas.height);
      // fill the representation background color (match the inner blue tone)
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0,0,canvas.width/ (window.devicePixelRatio||1), canvas.height/ (window.devicePixelRatio||1));
    });
    let currentPrompt = setNewPrompt();

    // init canvas background
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,canvas.width/ (window.devicePixelRatio||1), canvas.height/ (window.devicePixelRatio||1));

    // submit simulation: animate progress and set prediction text
    document.getElementById('prediction').addEventListener('click', async () => {

    const pred = document.getElementById('prediction');
    const pbar = document.getElementById('progressBar');

    pbar.style.transition = 'width 0.3s';
    pbar.style.width = '0%';
    pred.textContent = '...thinking';

    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
    const formData = new FormData();
    formData.append("image", blob, "drawing.png");

    try {
        const response = await fetch("http://127.0.0.1:8000/predict/", {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        console.log("Backend:", result);

        const predictedShape = result.prediction;
        const conf = result.score;

        pred.textContent = predictedShape.toUpperCase();
        pbar.style.transition = "width 1s ease";
        pbar.style.width = conf + "%";

        const correct = predictedShape.toLowerCase() === currentPrompt.toLowerCase();

    if (correct) {

        updateStars();
        showWinGif();
        currentPrompt = setNewPrompt();
        currentRound++;

        if (currentRound > maxRounds) {
            showWinPopup();
            popupCelebrate();
            return;
        }

        await resetRound();
    } else {
        const ratio = devicePixelRatio || 1;
        ctx.clearRect(0,0,canvas.width,canvas.height);
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0,0,canvas.width/ratio,canvas.height/ratio);

        pred.textContent = "TRY AGAIN!";
        pbar.style.width = "0%";

        showLoseGif(currentPrompt,predictedShape);
        const error = document.getElementById("errorSound");
        error.currentTime = 0;
        error.play().catch(e => console.log("Audio blocked:", e));

        canvas.style.pointerEvents = "auto";
    }

    } catch (err) {
        console.error(err);
        pred.textContent = "ERROR";
        pbar.style.width = "0%";
        alert("Prediction failed — is Django running?");
    }

  }); 

  document.getElementById("settingsBtn").addEventListener("click", () => {
    document.getElementById("settingsOverlay").style.display = "flex";
  });

  // CLOSE POPUP
  document.getElementById("closeSettings").addEventListener("click", () => {
      document.getElementById("settingsOverlay").style.display = "none";
  });

  // VOLUME CONTROL
  document.getElementById("bgVolume").addEventListener("input", (e) => {
      const v = e.target.value / 100;
      const bg = document.getElementById("bgMusic");
      bg.volume = v;
  });

  // MUTE/UNMUTE BUTTON
  document.getElementById("muteMusicBtn").addEventListener("click", () => {
      const bg = document.getElementById("bgMusic");

      if (bg.muted) {
          bg.muted = false;
          document.getElementById("muteMusicBtn").textContent = "MUTE MUSIC";
      } else {
          bg.muted = true;
          document.getElementById("muteMusicBtn").textContent = "UNMUTE MUSIC";
      }
  });

  document.getElementById("replayBtn").addEventListener("click", () => {
    
    // Hide popup
    hideWinPopup();

    // Reset game state
    currentRound = 1;
    starsEarned = 0;
    

    // Reset stars UI
    stars.forEach(s => s.classList.remove("earned"));

    // Reset prompt
    currentPrompt = setNewPrompt();
    showIdleGif(currentPrompt)

    // Clear canvas
    const ratio = devicePixelRatio || 1;
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0,0,canvas.width/ratio,canvas.height/ratio);

    // Reset prediction button + progress
    document.getElementById("prediction").textContent = "PREDICTION";
    document.getElementById("progressBar").style.width = "0%";

  });
  
    
    function showWinPopup() {
      const win = document.getElementById("winOverlay");
      win.style.display = "flex";
      win.style.pointerEvents = "auto";
    }

    function hideWinPopup() {
      const win = document.getElementById("winOverlay");
      win.style.display = "none";
      win.style.pointerEvents = "none"; 
    }

    function resetRound() {
      return new Promise(resolve => {
          canvas.style.pointerEvents = "none";

          setTimeout(() => {

              
              // Clear canvas
              ctx.clearRect(0, 0, canvas.width, canvas.height);
              ctx.fillStyle = "#ffffff";
              ctx.fillRect(
                  0,
                  0,
                  canvas.width / (devicePixelRatio || 1),
                  canvas.height / (devicePixelRatio || 1)
              );

              // Reset UI
              document.getElementById("prediction").textContent = "PREDICTION";
              document.getElementById("progressBar").style.width = "0%";

              canvas.style.pointerEvents = "auto";
              
              resolve();
          }, 1500);
        });
    }

    function popupCelebrate() {

      const winSound = document.getElementById("winSound");
      winSound.currentTime = 0; 
      winSound.play().catch(e => console.log("Audio blocked:", e));
      const canvas = document.getElementById("popupConfetti");

      const myConfetti = confetti.create(canvas, { resize: true });
      const duration = 500;
      const end = Date.now() + duration;

      (function frame() {
          myConfetti({
              particleCount: 8,
              spread: 60,
              origin: {
                  x: Math.random(),
                  y: Math.random()
              }
          });
          if (Date.now() < end) requestAnimationFrame(frame);
      })();
    }

    function updateStars() {
      if (starsEarned < stars.length) {
          stars[starsEarned].classList.add("earned");

          starsEarned++;
          const bonuspoint = document.getElementById("starEarned");
          bonuspoint.currentTime = 0; 
          bonuspoint.play().catch(e => console.log("Audio blocked:", e));
      }
    }

    function sleep(ms){return new Promise(r=>setTimeout(r,ms));}
    function randomPrediction(){
      const items=['CIRCLE','STAR','PENCIL','CAT','HOUSE','TREE','HEART'];
      return items[Math.floor(Math.random()*items.length)];
    }

    document.getElementById('shapesBtn').addEventListener('click', ()=>{
      // draw a demo star in the center
      const w = canvas.width/(window.devicePixelRatio||1);
      const h = canvas.height/(window.devicePixelRatio||1);
      drawStar(w/2, h/2, 30, 70, 5);
    });

    function drawStar(cx,cy,innerR,outerR,points){
      ctx.save();
      ctx.beginPath();
      for(let i=0;i<points*2;i++){
        const angle = Math.PI * i / points;
        const r = (i%2===0)?outerR:innerR;
        const x = cx + Math.cos(angle)*r;
        const y = cy + Math.sin(angle)*r;
        if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      ctx.closePath();
      ctx.fillStyle = '#ff6b3b';
      ctx.fill();
      ctx.restore();
    }

    document.querySelectorAll('.star').forEach(s=>s.addEventListener('click', (ev)=>{
      s.style.filter = s.style.filter? '': 'drop-shadow(0 4px 0 rgba(0,0,0,0.2))';
    }));

    window.addEventListener('resize', ()=>{
    });

    window.addEventListener("load", () => {
        showWelcomeGif();
    });

    function showWelcomeSequence() {
        setTimeout(() => {
            currentPrompt = setNewPrompt();
            showIdleGif();
    
            setTimeout(() => {
                const overlay = document.getElementById("welcomeOverlay");
                if (overlay) overlay.remove();
            }, 800);
    
        }, 3000);
    }
    
    window.addEventListener("load", showWelcomeSequence);