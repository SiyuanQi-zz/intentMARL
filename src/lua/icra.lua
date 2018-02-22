-- DO NOT WRITE CODE OUTSIDE OF THE if-then-end SECTIONS BELOW!! (unless the code is a function definition)

function initializeViews( )
    -- Parameters for city cameras
    local viewSizeX = 0.25
    local viewSizeY = 0.4

    local birdeyeCam = simGetObjectHandle('birdeye')
    local birdeyeView=simFloatingViewAdd(0.5*0.55, (1-viewSizeY)/2+viewSizeY, 0.55, 1-viewSizeY, 15)
    simAdjustView(birdeyeView,birdeyeCam,64)

    for i = 0, 3 do
        local cityCam = simGetObjectHandle('Camera'..i)
        local cityCamView=simFloatingViewAdd((i+0.5)*viewSizeX,0.5*viewSizeY,viewSizeX,viewSizeY,15)
        simAdjustView(cityCamView,cityCam,64)
    end

end

if (sim_call_type==sim_childscriptcall_initialization) then

    -- Put some initialization code here

    -- Make sure you read the section on "Accessing general-type objects programmatically"
    -- For instance, if you wish to retrieve the handle of a scene object, use following instruction:
    --
    -- handle=simGetObjectHandle('sceneObjectName')
    -- 
    -- Above instruction retrieves the handle of 'sceneObjectName' if this script's name has no '#' in it
    --
    -- If this script's name contains a '#' (e.g. 'someName#4'), then above instruction retrieves the handle of object 'sceneObjectName#4'
    -- This mechanism of handle retrieval is very convenient, since you don't need to adjust any code when a model is duplicated!
    -- So if the script's name (or rather the name of the object associated with this script) is:
    --
    -- 'someName', then the handle of 'sceneObjectName' is retrieved
    -- 'someName#0', then the handle of 'sceneObjectName#0' is retrieved
    -- 'someName#1', then the handle of 'sceneObjectName#1' is retrieved
    -- ...
    --
    -- If you always want to retrieve the same object's handle, no matter what, specify its full name, including a '#':
    --
    -- handle=simGetObjectHandle('sceneObjectName#') always retrieves the handle of object 'sceneObjectName' 
    -- handle=simGetObjectHandle('sceneObjectName#0') always retrieves the handle of object 'sceneObjectName#0' 
    -- handle=simGetObjectHandle('sceneObjectName#1') always retrieves the handle of object 'sceneObjectName#1'
    -- ...
    --
    -- Refer also to simGetCollisionhandle, simGetDistanceHandle, simGetIkGroupHandle, etc.
    --
    -- Following 2 instructions might also be useful: simGetNameSuffix and simSetNameSuffix

    -- **************** Parameters ****************
    sampleFrequency = 0.5 -- Hz
    humanLimit = 10
    sampleRange = math.floor(20/sampleFrequency)
    doorProb = 0.9

    unitReward = 1
    reward = {0, 0, 0}

    inspect = require('inspect')

    -- Get target handles for humans
    -- Set both bit 0 and bit 1 (3) to retrieve only the first children
    doorHandles = simGetObjectsInTree(simGetObjectHandle('doors'), sim_handle_all, 3)
    exitHandles = simGetObjectsInTree(simGetObjectHandle('exits'), sim_handle_all, 3)
    droneHandles = simGetObjectsInTree(simGetObjectHandle('drones'), sim_handle_all, 3)
    monitorHandles = simGetObjectsInTree(simGetObjectHandle('monitors'), sim_handle_all, 3)

    droneTargets = {-1, -1, -1}
    droneScriptHandles = {}
    for k, v in pairs(droneHandles) do
        droneScriptHandles[k] = simGetScriptAssociatedWithObject(v)
    end

    humanHandles = {}
    humanTargets = {}
    humanTargetAssigned = {}
    suspiciousPeople = {}
    visiblePeople = {}
    
    total_suspicious = 0
    total_captured = 0

    total_iteration = 0
    last_sampled_iteration = 0

    initializeViews()

    -- -- ********************** Testing **********************
    -- billHandle = simLoadModel('models/people/icraBillOMPL.ttm')
    -- -- simSetObjectPosition(billHandle, -1, {130, 53, 0})
    -- simSetObjectPosition(billHandle, -1, {190, 160, 0})
    -- for i=1,3 do
    --     droneTargets[i] = billHandle
    --     simCallScriptFunction('setTrackingTarget', droneScriptHandles[i], billHandle)
    -- end
    -- -- ********************** Testing **********************

    -- -- ********************** Case 1 **********************
    -- droneTargets[2] = simGetObjectHandle('door')
    -- simCallScriptFunction('setTrackingTarget', droneScriptHandles[2], simGetObjectHandle('door'))
    -- droneTargets[3] = simGetObjectHandle('door0')
    -- simCallScriptFunction('setTrackingTarget', droneScriptHandles[3], simGetObjectHandle('door0'))
    -- -- ********************** Case 1 **********************
end


function getRandomPosInTable(targetHandles, random)
    local targetHandle = targetHandles[math.random(#targetHandles)]
    local pos = simGetObjectPosition(targetHandle, -1)
    if random then
        return targetHandle, {pos[1]+math.random(-5, 5), pos[2]+math.random(-5, 5), pos[3]}
    else
        return targetHandle, pos
    end
end

function computeDistance(pt1, pt2)
    local squareSum = 0
    for k, v in pairs(pt1) do
        squareSum = squareSum + (pt1[k]-pt2[k])*(pt1[k]-pt2[k])
    end
    return math.sqrt(squareSum)
end

if (sim_call_type==sim_childscriptcall_actuation) then
    print(string.format('Iteration: %d, captured: %d/%d', total_iteration, total_captured, total_suspicious))
    math.randomseed(total_iteration)

    total_iteration = total_iteration + 1
    reward = {0, 0, 0}

    -- -- ********************** Testing **********************
    -- if total_iteration == 1 then
    --     local billScriptHandle = simGetScriptAssociatedWithObject(billHandle)
    --     simCallScriptFunction('setTarget', billScriptHandle, simGetObjectPosition(simGetObjectHandle('door'), -1))
    --     table.insert(humanHandles, billHandle)
    --     table.insert(humanTargets, simGetObjectHandle('door'))
    --     table.insert(humanTargetAssigned, true)
    --     table.insert(suspiciousPeople, true)
    --     table.insert(visiblePeople, true)
    -- end
    -- -- ********************** Testing **********************
    
    -- print(inspect(humanHandles), inspect(humanTargets), inspect(humanTargetAssigned), inspect(suspiciousPeople), inspect(visiblePeople))
    -- print(inspect(humanHandles), inspect(suspiciousPeople), inspect(visiblePeople))

    -- Find a target for the newly initialized human
    if #humanTargetAssigned > 0 and not humanTargetAssigned[#humanTargetAssigned] then
        local humanPos = simGetObjectPosition(humanHandles[#humanHandles], -1)
        local humanScriptHandle = simGetScriptAssociatedWithObject(humanHandles[#humanHandles])
        if simCallScriptFunction('checkPositionValid', humanScriptHandle, humanPos) then
            local targetHandle
            local targetPos = humanPos
            local suspicious
            -- print('Try to find a target for', humanHandles[#humanHandles])
            repeat
                targetPos = humanPos
                while computeDistance(targetPos, humanPos) < 0.001 do
                    -- print('while...')
                    if math.random() < doorProb then
                        targetHandle, targetPos = getRandomPosInTable(doorHandles, false)
                        suspicious = true
                    else
                        targetHandle, targetPos  = getRandomPosInTable(exitHandles, false)
                        suspicious = false
                    end
                end
            until simCallScriptFunction('setTarget', humanScriptHandle, targetPos)
            -- print('Target found!', inspect(targetPos), simCallScriptFunction('setTarget', humanScriptHandle, targetPos))
            humanTargetAssigned[#humanTargetAssigned] = true
            humanTargets[#humanTargets] = targetHandle
            table.insert(suspiciousPeople, suspicious)
        else
            local initHandle, initPos = getRandomPosInTable(exitHandles, true)
            simSetObjectPosition(humanHandles[#humanHandles], -1, initPos)
            last_sampled_iteration = total_iteration
        end
    else
        -- Sample a new human
        if #humanHandles <= humanLimit and total_iteration - last_sampled_iteration > sampleRange then
            if math.random(sampleRange) == 1 then
                local newHumanHandle = simLoadModel('models/people/icraBillOMPL.ttm')
                local humanScriptHandle = simGetScriptAssociatedWithObject(newHumanHandle)
                local initHandle, initPos = getRandomPosInTable(exitHandles, true)
                simSetObjectPosition(newHumanHandle, -1, initPos)
                table.insert(humanHandles, newHumanHandle)
                table.insert(humanTargetAssigned, false)
                table.insert(humanTargets, initHandle)
                last_sampled_iteration = total_iteration
            end
        end
    end

    -- Check if a human arrives his target
    -- If a suspicious human is observed entering a building, a reward is given
    local newHumanHandles = {}
    local newHumanTargets = {}
    local newhumanTargetAssigned = {}
    local newSuspiciousPeople = {}
    local newVisiblePeople = {}
    for k_human, v_human in pairs(humanHandles) do
        -- Check if any drones observe that
        visiblePeople[k_human] = {}
        for k_drone, v_drone in pairs(droneHandles) do
            if droneTargets[k_drone] == v_human or simCallScriptFunction('checkVisible', droneScriptHandles[k_drone], simGetObjectPosition(v_human, -1)) then
                table.insert(visiblePeople[k_human], k_drone)
            end
        end

        local humanScriptHandle = simGetScriptAssociatedWithObject(v_human)
        status = simCallScriptFunction('getStatus', humanScriptHandle)
        if status == 3 then
            -- If it is a target of a drone
            for k_drone_target, v_drone_target in pairs(droneTargets) do
                if v_human == v_drone_target then
                    droneTargets[k_drone_target] = humanTargets[k_human]
                    simCallScriptFunction('setTrackingTarget', droneScriptHandles[k_drone_target], humanTargets[k_human])
                end
            end

            -- If the human is suspicious, compute reward
            if suspiciousPeople[k_human] then
                total_suspicious  = total_suspicious + 1
                if #visiblePeople[k_human] > 0 then
                    total_captured = total_captured + 1
                    for k_visible_drone, v_visible_drone in pairs(visiblePeople[k_human]) do
                        reward[v_visible_drone] = reward[v_visible_drone] + unitReward/#visiblePeople[k_human]
                    end
                else
                    for k_drone, v_drone in pairs(droneHandles) do
                        reward[k_drone] = reward[k_drone] - unitReward/#droneHandles
                    end
                end

                -- if visiblePeople[k_human] then
                --     print('Captured suspicious human entered building!')
                --     total_captured = total_captured + 1
                --     if not reward then
                --         reward = unitReward
                --     else
                --         reward = reward + unitReward
                --     end
                -- else
                --     if not reward then
                --         reward = -unitReward
                --     else
                --         reward = reward - unitReward
                --     end
                -- end
            end

            simCallScriptFunction('cleanup', humanScriptHandle)
            simRemoveModel(v_human)
        else
            table.insert(newHumanHandles, v_human)
            table.insert(newHumanTargets, humanTargets[k_human])
            table.insert(newhumanTargetAssigned, humanTargetAssigned[k_human])
            table.insert(newSuspiciousPeople, suspiciousPeople[k_human])
            table.insert(newVisiblePeople, visiblePeople[k_human])
        end
    end
    humanHandles = newHumanHandles
    humanTargets = newHumanTargets
    humanTargetAssigned = newhumanTargetAssigned
    suspiciousPeople = newSuspiciousPeople
    visiblePeople = newVisiblePeople


    -- print('State:')
    -- print('Reward:', reward)
end


if (sim_call_type==sim_childscriptcall_sensing) then

    -- Put your main SENSING code here

end


if (sim_call_type==sim_childscriptcall_cleanup) then

    -- Put some restoration code here

end

-- For external calling
function getDrones(inInts,inFloats,inStrings,inBuffer)
    return droneHandles, {}, {}, ''
end

function getDoors(inInts,inFloats,inStrings,inBuffer)
    local outInts = {}
    local outFloats = {}
    for k, v in pairs(doorHandles) do
        outInts[k] = v
        local doorPos = simGetObjectPosition(v, -1)
        outFloats[2*k-1] = doorPos[1]
        outFloats[2*k] = doorPos[2]
    end
    return outInts, outFloats, {}, ''
end

function getExits(inInts,inFloats,inStrings,inBuffer)
    local outInts = {}
    local outFloats = {}
    for k, v in pairs(exitHandles) do
        outInts[k] = v
        local exitPos = simGetObjectPosition(v, -1)
        outFloats[2*k-1] = exitPos[1]
        outFloats[2*k] = exitPos[2]
    end
    return outInts, outFloats, {}, ''
end

function getMonitors(inInts,inFloats,inStrings,inBuffer)
    local outInts = {}
    local outFloats = {}
    for k, v in pairs(monitorHandles) do
        outInts[k] = v
        local monitorPos = simGetObjectPosition(v, -1)
        outFloats[2*k-1] = monitorPos[1]
        outFloats[2*k] = monitorPos[2]
    end
    return outInts, outFloats, {}, ''
end

function getState(inInts,inFloats,inStrings,inBuffer)
    local outInts = {}
    local outFloats = {}
    for k, v in pairs(droneHandles) do
        table.insert(outInts, v)
        table.insert(outInts, droneTargets[k])  -- Append target handle
        local dronePos = simGetObjectPosition(v, -1)
        table.insert(outFloats, dronePos[1])
        table.insert(outFloats, dronePos[2])
    end

    for k, v in pairs(visiblePeople) do
        if #v > 0 then
            table.insert(outInts, humanHandles[k])
            table.insert(outInts, humanTargets[k])  -- Append target handle
            -- table.insert
            local humanPos = simGetObjectPosition(humanHandles[k], -1)
            table.insert(outFloats, humanPos[1])
            table.insert(outFloats, humanPos[2])
        end
    end
    return outInts, outFloats, {}, ''
end

function getPerformance(inInts,inFloats,inStrings,inBuffer)
    return {total_captured, total_suspicious}, {}, {}, ''
end

function getReward(inInts,inFloats,inStrings,inBuffer)
    return {}, reward, {}, ''
end

function act(inInts,inFloats,inStrings,inBuffer)
    -- -- ********************** Case 1 **********************
    -- for k, v in pairs(inInts) do
    --     if k > 1 then
    --         break
    --     end
    --     if simIsHandleValid(v) == 1 then
    --         droneTargets[k] = v
    --         simCallScriptFunction('setTrackingTarget', droneScriptHandles[k], v)
    --     end
    -- end
    -- -- ********************** Case 1 **********************

    for k, v in pairs(inInts) do
        if simIsHandleValid(v) == 1 then
            droneTargets[k] = v
            simCallScriptFunction('setTrackingTarget', droneScriptHandles[k], v)
        end
    end
    return {}, {}, {}, ''
end