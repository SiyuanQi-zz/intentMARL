if (sim_call_type==sim_childscriptcall_initialization) then 
    -- Make sure we have version 2.4.13 or above (the particles are not supported otherwise)
    v=simGetInt32Parameter(sim_intparam_program_version)
    if (v<20413) then
        simDisplayDialog('Warning','The propeller model is only fully supported from V-REP version 2.4.13 and above.&&nThis simulation will not run as expected!',sim_dlgstyle_ok,false,'',nil,{0.8,0,0,0,0,0})
    end

    -- Detatch the manipulation sphere:
    targetObj=simGetObjectHandle('Quadricopter_target')
    simSetObjectParent(targetObj,-1,true)

    -- This control algo was quickly written and is dirty and not optimal. It just serves as a SIMPLE example

    d=simGetObjectHandle('Quadricopter_base')

    particlesAreVisible=simGetScriptSimulationParameter(sim_handle_self,'particlesAreVisible')
    simSetScriptSimulationParameter(sim_handle_tree,'particlesAreVisible',tostring(particlesAreVisible))
    simulateParticles=simGetScriptSimulationParameter(sim_handle_self,'simulateParticles')
    simSetScriptSimulationParameter(sim_handle_tree,'simulateParticles',tostring(simulateParticles))

    propellerScripts={-1,-1,-1,-1}
    for i=1,4,1 do
        propellerScripts[i]=simGetScriptHandle('Quadricopter_propeller_respondable'..i)
    end
    heli=simGetObjectAssociatedWithScript(sim_handle_self)

    particlesTargetVelocities={0,0,0,0}

    pParam=2
    iParam=0
    dParam=0
    vParam=-2

    cumul=0
    lastE=0
    pAlphaE=0
    pBetaE=0
    psp2=0
    psp1=0

    prevEuler=0


    fakeShadow=simGetScriptSimulationParameter(sim_handle_self,'fakeShadow')
    if (fakeShadow) then
        shadowCont=simAddDrawingObject(sim_drawing_discpoints+sim_drawing_cyclic+sim_drawing_25percenttransparency+sim_drawing_50percenttransparency+sim_drawing_itemsizes,0.2,0,-1,1)
    end

    -- Prepare 2 floating views with the camera views:
    floorCam=simGetObjectHandle('Quadricopter_floorCamera')
    frontCam=simGetObjectHandle('Quadricopter_frontCamera')
    floorView=simFloatingViewAdd(0.9,0.9,0.2,0.2,0)
    frontView=simFloatingViewAdd(0.7,0.9,0.2,0.2,0)
    simAdjustView(floorView,floorCam,64)
    simAdjustView(frontView,frontCam,64)
end 

if (sim_call_type==sim_childscriptcall_cleanup) then 
    simRemoveDrawingObject(shadowCont)
    simFloatingViewRemove(floorView)
    simFloatingViewRemove(frontView)
end 

if (sim_call_type==sim_childscriptcall_actuation) then 
    s=simGetObjectSizeFactor(d)
    
    pos=simGetObjectPosition(d,-1)
    if (fakeShadow) then
        itemData={pos[1],pos[2],0.002,0,0,1,0.2*s}
        simAddDrawingObjectItem(shadowCont,itemData)
    end
    
    -- Vertical control:
    targetPos=simGetObjectPosition(targetObj,-1)
    pos=simGetObjectPosition(d,-1)
    l=simGetVelocity(heli)
    e=(targetPos[3]-pos[3])
    cumul=cumul+e
    pv=pParam*e
    thrust=5.335+pv+iParam*cumul+dParam*(e-lastE)+l[3]*vParam
    lastE=e
    
    -- Horizontal control: 
    sp=simGetObjectPosition(targetObj,d)
    m=simGetObjectMatrix(d,-1)
    vx={1,0,0}
    vx=simMultiplyVector(m,vx)
    vy={0,1,0}
    vy=simMultiplyVector(m,vy)
    alphaE=(vy[3]-m[12])
    alphaCorr=0.25*alphaE+2.1*(alphaE-pAlphaE)
    betaE=(vx[3]-m[12])
    betaCorr=-0.25*betaE-2.1*(betaE-pBetaE)
    pAlphaE=alphaE
    pBetaE=betaE
    alphaCorr=alphaCorr+sp[2]*0.005+1*(sp[2]-psp2)
    betaCorr=betaCorr-sp[1]*0.005-1*(sp[1]-psp1)
    psp2=sp[2]
    psp1=sp[1]
    
    -- Rotational control:
    euler=simGetObjectOrientation(d,targetObj)
    rotCorr=euler[3]*0.1+2*(euler[3]-prevEuler)
    prevEuler=euler[3]
    
    -- Decide of the motor velocities:
    particlesTargetVelocities[1]=thrust*(1-alphaCorr+betaCorr+rotCorr)
    particlesTargetVelocities[2]=thrust*(1-alphaCorr-betaCorr-rotCorr)
    particlesTargetVelocities[3]=thrust*(1+alphaCorr-betaCorr+rotCorr)
    particlesTargetVelocities[4]=thrust*(1+alphaCorr+betaCorr-rotCorr)
    
    -- Send the desired motor velocities to the 4 rotors:
    for i=1,4,1 do
        simSetScriptSimulationParameter(propellerScripts[i],'particleVelocity',particlesTargetVelocities[i])
    end
end 
