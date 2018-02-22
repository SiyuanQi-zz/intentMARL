function computeDistance(pt1, pt2)
    local squareSum = 0
    for k, v in pairs(pt1) do
        squareSum = squareSum + (pt1[k]-pt2[k])*(pt1[k]-pt2[k])
    end
    return math.sqrt(squareSum)
end

function computeRelativeOrientation(currentPos, targetPos)
    local ori
    if targetPos[1]-currentPos[1] >= 0 then
        ori = {0, 0, math.atan((targetPos[2]-currentPos[2])/(targetPos[1]-currentPos[1]))}
    else
        ori = {0, 0, math.pi+math.atan((targetPos[2]-currentPos[2])/(targetPos[1]-currentPos[1]))}
    end
    return ori
end

function checkPositionValid(inputPosition)
    -- print('check validity!')
    -- local goalPostition
    -- if #inputPosition == 2 then
    --     goalPostition = inputPosition
    -- elseif #inputPosition == 3 then
    --     goalPostition = {inputPosition[1], inputPosition[2]}
    -- else
    --     print('Target dimention incorrect!')
    --     return false
    -- end

    -- local goalPostition = inputPosition

    -- if not simExtOMPL_isStateValid(taskHandle, inputPosition) == 1 then
    --     return false
    -- end
    local goalPostition = {inputPosition[1], inputPosition[2], droneHeight}

    -- print(targetHandle, inspect(stateSpaceHandle), inspect(inputPosition), inspect(goalPostition))
    -- print(heliSuffix, inspect(droneHandles))
    if simExtOMPL_isStateValid(taskHandle, inputPosition) ~= 1 then
        return false
    end

    -- Check collision with other drones
    for k, v in pairs(droneHandles) do
        if k ~= (heliSuffix+2) and computeDistance(goalPostition, simGetObjectPosition(v, -1)) < 1 then
            return false
        end
    end

    -- Check if the path is valid
    local currentTargetPos = simGetObjectPosition(targetObj, -1)
    local r, path = computePath({currentTargetPos[1], currentTargetPos[2]}, inputPosition)
    if r == 0 then
        return false
    end

    -- -- Check collision with buildings
    -- local currentTargetPos = simGetObjectPosition(targetObj, -1)
    -- simSetObjectPosition(targetObj, -1, goalPostition)
    -- for k, v in pairs(obstacleHandles) do
    --     print('Obstacle', simGetObjectName(v))

    --     local r, dis = simCheckDistance(targetObj, v, 5)
    --     if r==1 then
    --         print('distance', simGetObjectName(v), inspect(goalPostition), simCheckCollision(targetObj, v), inspect(dis))
    --     end

    --     if simCheckCollision(targetObj, v) == 1 then
    --         simSetObjectPosition(targetObj, -1, currentTargetPos)
    --         return false
    --     end
    -- end
    -- simSetObjectPosition(targetObj, -1, currentTargetPos)

    -- local r, tag, locLow, locHigh = simCheckOctreePointOccupancy(simGetObjectHandle('Octree'), 0, goalPostition)
    -- print('octree check', inspect(goalPostition), r, tag, locLow, locHigh)
    -- if r == 1 then
    --     return false
    -- end

    return true
end

function setTargetPos(pos)
    -- ********** Using OMPL
    simSetObjectPosition(targetObj, -1, pos)
    -- ********** Using OMPL

    -- -- ********** Using acceleration
    -- totalTargetTargetDistance = 0
    -- currentTargetTargetProgress = 0
    -- targetV = {0, 0}
    -- targetDirection = {0, 0}
    -- simSetObjectPosition(targetObj, -1, pos)
    -- -- ********** Using acceleration
end

function setTargetTarget(pos, ori)
    -- ********** Using OMPL
    targetPath = {}
    targetTargetPosition = pos
    pathCalculated = 0
    currentIndexOnPath = 1

    -- -- Do the path planning by OMPL
    -- local currentTargetPos = simGetObjectPosition(targetObj,-1)
    -- local r, path = computePath({currentTargetPos[1], currentTargetPos[2]}, {targetTargetPosition[1], targetTargetPosition[2]})
    -- if r ~= 0 then
    --     targetPath = path
    --     visualize2dPath(targetPath, currentTargetPos[3])
    --     pathCalculated = 1
    -- end
    -- ********** Using OMPL

    -- -- ********** Using acceleration
    -- local targetTargetPosition = pos
    -- local targetPos = simGetObjectPosition(targetObj, -1)
    -- totalTargetTargetDistance = computeDistance(targetPos, targetTargetPosition)
    -- if totalTargetTargetDistance > targetTargetDisThresh then
    --     currentTargetTargetProgress = 0
    --     targetV = {0, 0}
    --     targetDirection = {(pos[1]-targetPos[1])/totalTargetTargetDistance, (pos[2]-targetPos[2])/totalTargetTargetDistance}
    -- elseif totalTargetTargetDistance > 0 then
    --     setTargetPos(pos)
    -- end
    -- -- ********** Using acceleration

    -- simSetObjectOrientation(targetObj, -1, ori)
end

function setTrackingTarget(objHandle)
    if objHandle ~= trackingTargetHandle then
        trackingTargetHandle = objHandle
        if objHandle == -1 then
            lastTrackingTargetPos = nil
        else
            pathCalculated = 3
        end
    end
end

function checkVisible(position)
    position[3] = 0.8 -- For human tracking

    local m = simGetObjectMatrix(frontSensor,-1)
    m = simGetInvertedMatrix(m)
    local position_camera = simMultiplyVector(m,position)

    if position_camera[3] < 0 then
        return false
    end

    local currentAngleX = math.atan(position_camera[1] / position_camera[3])
    local currentAngleY = math.atan(position_camera[2] / position_camera[3])
    if math.abs(currentAngleX) < halfAngleX and math.abs(currentAngleY) < halfAngleY then
        return true
    else
        return false
    end

end

function getObjectPositionFromSensor(objHandle)
    local position = simGetObjectPosition(objHandle, -1)
    if checkVisible(position) then
        return position
    else
        return nil
    end
end

function visualize2dPath(path, height)
    initPos = {path[1], path[2]}
    if not _lineContainer then
        _lineContainer=simAddDrawingObject(sim_drawing_lines,3,0,-1,99999,{0.2,0.2,0.2})
    end
    simAddDrawingObjectItem(_lineContainer,nil)
    if path then
        local pc=#path/2
        for i=1, pc-1 do
            lineDat={path[(i-1)*2+1],path[(i-1)*2+2],height,path[i*2+1],path[i*2+2],height}
            simAddDrawingObjectItem(_lineContainer,lineDat)
        end
    end
end

function computePath(desiredInitPos, desiredTargetPos)
    -- if not (checkPositionValid(desiredInitPos) and checkPositionValid(desiredTargetPos)) then
    --     return 0, {}
    -- end

    -- print('computing path for', inspect(desiredInitPos), inspect(desiredTargetPos))
    -- desiredInitPos[3] = 8
    -- desiredTargetPos[3] = 8
    local maxTime = 0.5
    -- local minStates = 100

    simExtOMPL_setStartState(taskHandle, desiredInitPos)
    simExtOMPL_setGoalState(taskHandle, desiredTargetPos)
    -- local r, path = simExtOMPL_compute(taskHandle, maxTime, -1, minStates) -- r = 0 if not successful
    local r, path = simExtOMPL_compute(taskHandle, maxTime) -- r = 0 if not successful
    return r, path
end

function updateTarget()
    -- ********** Using OMPL
    -- print('Current index on path', currentIndexOnPath)
    -- print('pathCalculated', pathCalculated)

    if pathCalculated==0 then
        local currentTargetPos = simGetObjectPosition(targetObj,-1)
        -- print('compute path for', inspect(targetTargetPosition))
        -- print('updateTarget: not calculated', inspect(targetTargetPosition), checkPositionValid({targetTargetPosition[1], targetTargetPosition[2]}))
        local r, path = computePath({currentTargetPos[1], currentTargetPos[2]}, {targetTargetPosition[1], targetTargetPosition[2]})
        if r ~= 0 then
            targetPath = path
            -- visualize2dPath(targetPath, currentTargetPos[3])
            pathCalculated = 1
        end
    end

    if pathCalculated==1 then
        if currentIndexOnPath < #targetPath-2 then
            local currentTargetPos = simGetObjectPosition(targetObj,-1)
            local currentTargetPos2d = {currentTargetPos[1], currentTargetPos[2]}

            while true do
                local nextPos = {targetPath[currentIndexOnPath], targetPath[currentIndexOnPath+1]}
                if computeDistance(currentTargetPos2d, nextPos) > droneSpeed or currentIndexOnPath >= #targetPath-2 then
                    -- if checkPositionValid({targetPath[currentIndexOnPath], targetPath[currentIndexOnPath+1]}) then
                        -- break
                    -- end
                    break
                end
                currentIndexOnPath = currentIndexOnPath + 2
            end

            -- print(inspect({targetPath[currentIndexOnPath], targetPath[currentIndexOnPath+1], currentTargetPos[3]}))
            setTargetPos({targetPath[currentIndexOnPath], targetPath[currentIndexOnPath+1], currentTargetPos[3]})
            simSetObjectOrientation(targetObj, -1, computeRelativeOrientation(simGetObjectPosition(targetObj,-1), lastTrackingTargetPos))
        else
            pathCalculated = 2
        end
    elseif pathCalculated==2 then
        setTargetPos(targetTargetPosition)
        simSetObjectOrientation(targetObj, -1, computeRelativeOrientation(simGetObjectPosition(targetObj,-1), lastTrackingTargetPos))
        pathCalculated = 3
    else
        -- idle
    end
    -- ********** Using OMPL

    -- -- ********** Using acceleration
    -- if totalTargetTargetDistance > 0 then
    --     if currentTargetTargetProgress <= totalTargetTargetDistance/2 then
    --         -- targetV = {targetDirection[1]*targetAcceleration, targetDirection[2]*targetAcceleration}
    --         targetV = {targetV[1]+targetDirection[1]*targetAcceleration, targetV[2]+targetDirection[2]*targetAcceleration}
    --     elseif currentTargetTargetProgress <= totalTargetTargetDistance then
    --         -- targetV = {targetDirection[1]*targetAcceleration, targetDirection[2]*targetAcceleration}
    --         targetV = {targetV[1]-targetDirection[1]*targetAcceleration, targetV[2]-targetDirection[2]*targetAcceleration}
    --     else
    --         -- Already pass target
    --         setTargetPos(pos)
    --     end


    --     local targetPos = simGetObjectPosition(targetObj, -1)
    --     simSetObjectPosition(targetObj, -1, {targetPos[1]+targetV[1], targetPos[2]+targetV[2], targetPos[3]})
    --     currentTargetTargetProgress = currentTargetTargetProgress + math.sqrt(targetV[1]*targetV[1]+targetV[2]*targetV[2])
    -- end
    -- -- ********** Using acceleration
end

function createPlanningSpace( )
    targetPath = {}
    targetTargetPosition = simGetObjectPosition(targetObj, -1)
    pathCalculated = 3 -- 0=not calculated, 1=calculated, 2=completed, 3=waiting
    currentIndexOnPath=1

    taskHandle = simExtOMPL_createTask('droneTargetTask')
    stateSpaceHandle = {simExtOMPL_createStateSpace('droneStateSpace', sim_ompl_statespacetype_position2d, heli, {0, 0}, {228, 213}, 1)}
    -- stateSpaceHandle = {simExtOMPL_createStateSpace('droneStateSpace', sim_ompl_statespacetype_position2d, heli, {-25, -25}, {25, 25}, 1)} -- For testing
    simExtOMPL_setStateValidityCheckingResolution(taskHandle, 0.001)
    droneAppoxVol = simGetCollectionHandle('drone_approxVolume')
    droneObstacles = simGetCollectionHandle('drone_obstacles')
    simExtOMPL_setStateSpace(taskHandle, stateSpaceHandle)
    simExtOMPL_setAlgorithm(taskHandle, sim_ompl_algorithm_BiTRRT)
    simExtOMPL_setCollisionPairs(taskHandle, {droneAppoxVol, droneObstacles})

    simExtOMPL_setVerboseLevel(taskHandle, 0) -- 0 to suppress any message
end


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

    -- Tracking purpose
    local alpha = 45
    local objectHeight = 0.8
    droneHeight = simGetObjectPosition(heli, -1)[3]

    inspect = require('inspect')
    trackingRadius = (droneHeight - objectHeight) * math.tan(alpha*math.pi/180.0)
    lastTrackingTargetPos = nil
    trackingTargetHandle = -1

    -- Compute path for the target itself
    droneSpeed = 1  -- 1m/50ms is 44.7387mph
    -- ********** Using OMPL
    createPlanningSpace()
    local dronePos = simGetObjectPosition(heli, -1)
    simExtOMPL_setStartState(taskHandle, {dronePos[1], dronePos[2]})
    simExtOMPL_setGoalState(taskHandle, {dronePos[1]+math.random(), dronePos[2]+math.random()})
    local r, path = simExtOMPL_compute(taskHandle, 0.01) -- r = 0 if not successful
    -- ********** Using OMPL

    -- -- ********** Using acceleration
    -- targetTargetDisThresh = 0.1
    -- targetAcceleration = 0.1
    -- totalTargetTargetDistance = 0
    -- currentTargetTargetProgress = 0
    -- targetV = {0, 0}
    -- targetDirection = {0, 0}
    -- -- ********** Using acceleration


    -- Handle multiple drones
    droneHandles = simGetObjectsInTree(simGetObjectHandle('drones#'), sim_handle_all, 3)
    obstacleHandles = simGetObjectsInTree(simGetObjectHandle('Obstacles#'), sim_handle_all, 3)
    heliSuffix = simGetNameSuffix(simGetObjectName(heli))
    totalDrones = #droneHandles
    -- totalDrones = 0
    -- local allObjHandles = simGetObjectsInTree(sim_handle_scene, sim_handle_all, 3)
    -- for i=1,#allObjHandles do
    --     -- print(simGetObjectName(allObjHandles[i]))
    --     if string.sub(simGetObjectName(allObjHandles[i]),1,12)=='Quadricopter' then
    --         totalDrones = totalDrones + 1
    --     end
    -- end
    -- totalDrones = totalDrones-(heliSuffix+2)  -- Detached targets are duplicated

    -- Prepare 2 floating views with the camera views:
    -- floorCam=simGetObjectHandle('Quadricopter_floorCamera')
    -- frontCam=simGetObjectHandle('Quadricopter_frontCamera')
    resolutionX = 640
    resolutionY = 480
    mapResolutionX = 213
    mapResolutionY = 228

    frontSensor=simGetObjectHandle('Quadricopter_frontVisionSensor')
    passiveDetectSensor=simGetObjectHandle('Quadricopter_passiveDetectVisionSensor')
    passiveMapSensor=simGetObjectHandle('Quadricopter_passiveMapVisionSensor')

    perspectiveAngle = 60*math.pi/180
    -- perspectiveAngle = simGetObjectFloatParameter(frontSensor, 1004)
    local ratio=resolutionX/resolutionY
    if (ratio>1) then
        halfAngleX = perspectiveAngle/2
        halfAngleY = math.atan(math.tan(perspectiveAngle/2)/ratio)
    else
        halfAngleX = math.atan(math.tan(perspectiveAngle/2)*ratio)
        halfAngleY = perspectiveAngle/2
    end

    simSetObjectInt32Parameter(frontSensor, 1002, resolutionX)
    simSetObjectInt32Parameter(frontSensor, 1003, resolutionY)
    simSetObjectInt32Parameter(passiveDetectSensor, 1002, mapResolutionX)
    simSetObjectInt32Parameter(passiveDetectSensor, 1003, mapResolutionY)
    simSetObjectInt32Parameter(passiveMapSensor, 1002, mapResolutionX)
    simSetObjectInt32Parameter(passiveMapSensor, 1003, mapResolutionY)

    simSetObjectFloatParameter(frontSensor, 1004, perspectiveAngle)
    simSetObjectFloatParameter(passiveDetectSensor, 1004, perspectiveAngle)
    simSetObjectFloatParameter(passiveMapSensor, 1004, perspectiveAngle)

    local viewSizeX = 0.15
    local viewSizeY = 0.2
    frontView=simFloatingViewAdd(1-2.5*viewSizeX,1-0.5*viewSizeY-viewSizeY*(heliSuffix+1),viewSizeX,viewSizeY,15)
    detectView=simFloatingViewAdd(1-1.5*viewSizeX,1-0.5*viewSizeY-viewSizeY*(heliSuffix+1),viewSizeX,viewSizeY,15)
    mapView=simFloatingViewAdd(1-0.5*viewSizeX,1-0.5*viewSizeY-viewSizeY*(heliSuffix+1),viewSizeX,viewSizeY,15)
    simAdjustView(frontView,frontSensor,64)
    simAdjustView(detectView,passiveDetectSensor,64)
    simAdjustView(mapView,passiveMapSensor,64)
    
    -- Enable an image publisher and subscriber:
    publisher=simExtRosInterface_advertise('/image', 'sensor_msgs/Image')
    simExtRosInterface_publisherTreatUInt8ArrayAsString(publisher) -- treat uint8 arrays as strings (much faster, tables/arrays are kind of slow in Lua)

    detectSubscriber=simExtRosInterface_subscribe('/detection', 'sensor_msgs/Image', 'detectionMessage_callback', totalDrones)
    simExtRosInterface_subscriberTreatUInt8ArrayAsString(detectSubscriber) -- treat uint8 arrays as strings (much faster, tables/arrays are kind of slow in Lua)


    mapSubscriber=simExtRosInterface_subscribe('/map', 'sensor_msgs/Image', 'mapMessage_callback', totalDrones)
    simExtRosInterface_subscriberTreatUInt8ArrayAsString(mapSubscriber)
end 

if (sim_call_type==sim_childscriptcall_cleanup) then 
    simSetObjectParent(targetObj,heli,true)

    simRemoveDrawingObject(shadowCont)
    simFloatingViewRemove(frontView)
    simFloatingViewRemove(detectView)
    simFloatingViewRemove(mapView)

    -- Shut down publisher and subscriber. Not really needed from a simulation script (automatic shutdown)
    simExtRosInterface_shutdownPublisher(publisher)
    simExtRosInterface_shutdownSubscriber(detectSubscriber)
    simExtRosInterface_shutdownSubscriber(mapSubscriber)
end 


function track()
    local currentPos = simGetObjectPosition(targetObj, -1)
    local targetPos = simGetObjectPosition(trackingTargetHandle, -1)

    -- local dis = computeDistance({currentPos[1], currentPos[2]}, {targetPos[1], targetPos[2]})
    -- local pos = {(currentPos[1]-targetPos[1])*trackingRadius/dis+targetPos[1], (currentPos[2]-targetPos[2])*trackingRadius/dis+targetPos[2], droneHeight}

    -- local ori
    -- if targetPos[1]-currentPos[1] >= 0 then
    --     ori = {0, 0, math.atan((targetPos[2]-currentPos[2])/(targetPos[1]-currentPos[1]))}
    -- else
    --     ori = {0, 0, math.pi+math.atan((targetPos[2]-currentPos[2])/(targetPos[1]-currentPos[1]))}
    -- end

    local theta
    if currentPos[1]-targetPos[1] >= 0 then
        theta = math.atan((currentPos[2]-targetPos[2])/(currentPos[1]-targetPos[1]))
    else
        theta = math.pi+math.atan((currentPos[2]-targetPos[2])/(currentPos[1]-targetPos[1]))
    end

    local pos, ori
    local divideAngle = 24
    for i = 0,divideAngle-1 do
        pos = {targetPos[1]+trackingRadius*math.cos(theta+i*2*math.pi/divideAngle), targetPos[2]+trackingRadius*math.sin(theta+i*2*math.pi/divideAngle), droneHeight}
        if checkPositionValid({pos[1], pos[2]}) then
            break
            -- local r, path = computePath({currentPos[1], currentPos[2]}, {pos[1], pos[2]})
            -- if r~=0 then
            --     break
            -- end
        end
    end
    if pos then
        ori = computeRelativeOrientation(pos, targetPos)
    end

    return pos, ori
end

function detectionMessage_callback(msg)
    -- Apply the received image to the passive vision sensor that acts as an image container
    if tonumber(msg.header.frame_id) == heliSuffix then
        simSetVisionSensorCharImage(passiveDetectSensor,msg.data)
    end
end

function mapMessage_callback(msg)
    -- Apply the received image to the passive vision sensor that acts as an image container
    if tonumber(msg.header.frame_id) == heliSuffix then
        simSetVisionSensorCharImage(passiveMapSensor,msg.data)
    end
end

function publishImage()
    -- Publish the image of the active vision sensor:
    local heliSuffix = simGetNameSuffix(simGetObjectName(heli))
    local data,w,h=simGetVisionSensorCharImage(frontSensor)
    local imageData={}
    imageData['header']={seq=0,stamp=simExtRosInterface_getTime(), frame_id=tostring(heliSuffix)}
    imageData['height']=h
    imageData['width']=w
    imageData['encoding']='rgb8'
    imageData['is_bigendian']=1
    imageData['step']=w*3
    imageData['data']=data
    simExtRosInterface_publish(publisher,imageData)
end

if (sim_call_type==sim_childscriptcall_sensing) then
    -- Handle vision sensors
    publishImage()
end


if (sim_call_type==sim_childscriptcall_actuation) then
    -- print('updateTarget begin', heliSuffix)
    updateTarget()
    -- print('updateTarget end', heliSuffix)

    -- Handle tracking
    if pathCalculated == 3 then
        if trackingTargetHandle ~= -1 then
            local trackingTargetPos = simGetObjectPosition(trackingTargetHandle, -1)
            if lastTrackingTargetPos == nil or computeDistance(trackingTargetPos, lastTrackingTargetPos) > 3 then
                lastTrackingTargetPos = trackingTargetPos
                local newTargetTargetPos, newTargetTargetOri = track()
                if newTargetTargetPos then
                    -- print('handle tracking: setTargetTarget', inspect(newTargetTargetPos))
                    setTargetTarget(newTargetTargetPos, newTargetTargetOri)
                end
            end
        end
    end

    -- Move drone to target position if the current position is not valid
    -- print('Checking heli valid', pathCalculated)
    if pathCalculated == 0 then
        local heliPos = simGetObjectPosition(heli, -1)
        if not checkPositionValid({heliPos[1], heliPos[2]}) then
            -- print('Heli position not valid!', heliSuffix)
            local newTargetTargetPos, newTargetTargetOri = track()
            if newTargetTargetPos then
                -- setTargetTarget(newTargetTargetPos, newTargetTargetOri)
                pathCalculated = 3
                simSetObjectPosition(targetObj, -1, newTargetTargetPos)
                simSetObjectOrientation(targetObj, -1, newTargetTargetOri)
            end
            -- simSetObjectPosition(targetObj, -1, targetTargetPosition)
        end
    end
    -- print('heli position checked')


    -- Testing target movement
    simSetObjectPosition(heli, -1, simGetObjectPosition(targetObj, -1))
    simSetObjectOrientation(heli, -1, simGetObjectOrientation(targetObj, -1))

    -- Original quadricopter control code
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
