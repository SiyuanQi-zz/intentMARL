------------------------------------------------------------------------------ 
-- Following few lines automatically added by V-REP to guarantee compatibility 
-- with V-REP 3.1.3 and earlier: 
colorCorrectionFunction=function(_aShapeHandle_) 
  local version=simGetInt32Parameter(sim_intparam_program_version) 
  local revision=simGetInt32Parameter(sim_intparam_program_revision) 
  if (version<30104)and(revision<3) then 
      return _aShapeHandle_ 
  end 
  return '@backCompatibility1:'.._aShapeHandle_ 
end 
------------------------------------------------------------------------------ 
 
 
setColor=function(objectTable,colorName,color)
    for i=1,#objectTable,1 do
        if (simGetObjectType(objectTable[i])==sim_object_shape_type) then
            simSetShapeColor(colorCorrectionFunction(objectTable[i]),colorName,0,color)
        end
    end
end

setTarget=function(position)
    simSetObjectPosition(targetHandle, -1, position)
end

if (sim_call_type==sim_childscriptcall_initialization) then 
    BillHandle=simGetObjectHandle('Bill')
    legJointHandles={simGetObjectHandle('Bill_leftLegJoint'),simGetObjectHandle('Bill_rightLegJoint')}
    kneeJointHandles={simGetObjectHandle('Bill_leftKneeJoint'),simGetObjectHandle('Bill_rightKneeJoint')}
    ankleJointHandles={simGetObjectHandle('Bill_leftAnkleJoint'),simGetObjectHandle('Bill_rightAnkleJoint')}
    shoulderJointHandles={simGetObjectHandle('Bill_leftShoulderJoint'),simGetObjectHandle('Bill_rightShoulderJoint')}
    elbowJointHandles={simGetObjectHandle('Bill_leftElbowJoint'),simGetObjectHandle('Bill_rightElbowJoint')}
    
    pathHandle=simGetObjectHandle('Bill_path')
    targetHandle=simGetObjectHandle('Bill_goalDummy')
    pathPlanningHandle=simGetPathPlanningHandle('Bill_task')
    
    simSetObjectParent(pathHandle,-1,true)
    simSetObjectParent(targetHandle,-1,true)
    
    legWaypoints={0.237,0.228,0.175,-0.014,-0.133,-0.248,-0.323,-0.450,-0.450,-0.442,-0.407,-0.410,-0.377,-0.303,-0.178,-0.111,-0.010,0.046,0.104,0.145,0.188}
    kneeWaypoints={0.282,0.403,0.577,0.929,1.026,1.047,0.939,0.664,0.440,0.243,0.230,0.320,0.366,0.332,0.269,0.222,0.133,0.089,0.065,0.073,0.092}
    ankleWaypoints={-0.133,0.041,0.244,0.382,0.304,0.232,0.266,0.061,-0.090,-0.145,-0.043,0.041,0.001,0.011,-0.099,-0.127,-0.121,-0.120,-0.107,-0.100,-0.090,-0.009}
    shoulderWaypoints={0.028,0.043,0.064,0.078,0.091,0.102,0.170,0.245,0.317,0.337,0.402,0.375,0.331,0.262,0.188,0.102,0.094,0.086,0.080,0.051,0.058,0.048}
    elbowWaypoints={-1.148,-1.080,-1.047,-0.654,-0.517,-0.366,-0.242,-0.117,-0.078,-0.058,-0.031,-0.001,-0.009,0.008,-0.108,-0.131,-0.256,-0.547,-0.709,-0.813,-1.014,-1.102}
    relativeVel={2,2,1.2,2.3,1.4,1,1,1,1,1.6,1.9,2.4,2.0,1.9,1.5,1,1,1,1,1,2.3,1.5}
    
    nominalVelocity=simGetScriptSimulationParameter(sim_handle_self,'walkingSpeed')
    randomColors=simGetScriptSimulationParameter(sim_handle_self,'randomColors')
    scaling=0
    tl=#legWaypoints
    dl=1/tl
    vp=0
    desiredTargetPos={-99,-99}
    pathCalculated=0 -- 0=not calculated, 1=beeing calculated, 2=calculated
    tempPathSearchObject=-1
    currentPosOnPath=0
    
    HairColors={4,{0.30,0.22,0.14},{0.75,0.75,0.75},{0.075,0.075,0.075},{0.75,0.68,0.23}}
    skinColors={2,{0.61,0.54,0.45},{0.52,0.45,0.35}}
    shirtColors={5,{0.27,0.36,0.54},{0.54,0.27,0.27},{0.31,0.51,0.33},{0.46,0.46,0.46},{0.18,0.18,0.18}}
    trouserColors={2,{0.4,0.34,0.2},{0.12,0.12,0.12}}
    shoeColors={2,{0.12,0.12,0.12},{0.25,0.12,0.045}}
    
    -- Initialize to random colors if desired:
    if (randomColors) then
        -- First we just retrieve all objects in the model:
        previousSelection=simGetObjectSelection()
        simRemoveObjectFromSelection(sim_handle_all,-1)
        simAddObjectToSelection(sim_handle_tree,BillHandle)
        modelObjects=simGetObjectSelection()
        simRemoveObjectFromSelection(sim_handle_all,-1)
        simAddObjectToSelection(previousSelection)
        -- Now we set random colors:
        math.randomseed(simGetFloatParameter(sim_floatparam_rand)*10000) -- each lua instance should start with a different and 'good' seed
        setColor(modelObjects,'HAIR',HairColors[1+math.random(HairColors[1])])
        setColor(modelObjects,'SKIN',skinColors[1+math.random(skinColors[1])])
        setColor(modelObjects,'SHIRT',shirtColors[1+math.random(shirtColors[1])])
        setColor(modelObjects,'TROUSERS',trouserColors[1+math.random(trouserColors[1])])
        setColor(modelObjects,'SHOE',shoeColors[1+math.random(shoeColors[1])])
    end
end 

if (sim_call_type==sim_childscriptcall_cleanup) then 
    simSetObjectParent(pathHandle,BillHandle,true)
    simSetObjectParent(targetHandle,BillHandle,true)
    -- Restore to initial colors:
    if (randomColors) then
        previousSelection=simGetObjectSelection()
        simRemoveObjectFromSelection(sim_handle_all,-1)
        simAddObjectToSelection(sim_handle_tree,BillHandle)
        modelObjects=simGetObjectSelection()
        simRemoveObjectFromSelection(sim_handle_all,-1)
        simAddObjectToSelection(previousSelection)
        setColor(modelObjects,'HAIR',HairColors[2])
        setColor(modelObjects,'SKIN',skinColors[2])
        setColor(modelObjects,'SHIRT',shirtColors[2])
        setColor(modelObjects,'TROUSERS',trouserColors[2])
        setColor(modelObjects,'SHOE',shoeColors[2])
    end
end 

if (sim_call_type==sim_childscriptcall_actuation) then 
    s=simGetObjectSizeFactor(BillHandle)
    
    -- Check if we need to recompute the path (e.g. because the goal position has moved):
    targetP=simGetObjectPosition(targetHandle,-1)
    vv={targetP[1]-desiredTargetPos[1],targetP[2]-desiredTargetPos[2]}
    if (math.sqrt(vv[1]*vv[1]+vv[2]*vv[2])>0.01) then
        pathCalculated=0 -- We have to recompute the path since the target position has moved
        desiredTargetPos[1]=targetP[1]
        desiredTargetPos[2]=targetP[2]
    end
    
    rightV=0
    leftV=0
    
        if (pathCalculated==0) then
            -- We need to initialize a path search object:
            if (tempPathSearchObject~=-1) then
                -- delete any previous temporary path search object:    
                simPerformPathSearchStep(tempPathSearchObject,true) 
            end
            tempPathSearchObject=simInitializePathSearch(pathPlanningHandle,5,0.03) -- search for a maximum of 5 seconds
            if (tempPathSearchObject~=-1) then
                pathCalculated=1 -- Initialization went fine
            end
        else
            if (pathCalculated==1) then
                -- A path hasn't been found yet, we need to perform another path search step:
                r=simPerformPathSearchStep(tempPathSearchObject,false)
                if (r<1) then
                    -- Path was not yet found, or the search has failed
                    if (r~=-2) then
                        -- path search has failed!
                        pathCalculated=0
                        tempPathSearchObject=-1
                    end
                else
                    -- we found a path!
                    pathCalculated=2 
                    currentPosOnPath=0
                    tempPathSearchObject=-1
                end
            else
                -- We have an existing path. We follow that path:
                l=simGetPathLength(pathHandle)
                r=simGetObjectPosition(BillHandle,-1)
                while true do
                    p=simGetPositionOnPath(pathHandle,currentPosOnPath/l)
                    d=math.sqrt((p[1]-r[1])*(p[1]-r[1])+(p[2]-r[2])*(p[2]-r[2]))
                    if (d>0.3)or(currentPosOnPath>=l) then
                        break
                    end
                    currentPosOnPath=currentPosOnPath+0.05
                end
                if (d>0.1) then
                    -- Ok, we follow the path
                    m=simGetObjectMatrix(BillHandle,-1)
                    m=simGetInvertedMatrix(m)
                    p=simMultiplyVector(m,p)
                    -- Now p is relative to the mannequin
                    a=math.atan2(p[2],p[1])
                    if (a>=0)and(a<math.pi*0.5) then
                        rightV=nominalVelocity
                        leftV=nominalVelocity*(1-2*a/(math.pi*0.5))
                    end
                    if (a>=math.pi*0.5) then
                        leftV=-nominalVelocity
                        rightV=nominalVelocity*(1-2*(a-math.pi*0.5)/(math.pi*0.5))
                    end
                    if (a<0)and(a>-math.pi*0.5) then
                        leftV=nominalVelocity
                        rightV=nominalVelocity*(1+2*a/(math.pi*0.5))
                    end
                    if (a<=-math.pi*0.5) then
                        rightV=-nominalVelocity
                        leftV=nominalVelocity*(1+2*(a+math.pi*0.5)/(math.pi*0.5))
                    end
                else
                    -- We arrived at the end of the path. The position of Bill still might not
                    -- coincide with the goal position if we selected the "partial path" option in the path planning dialog
                    targetP=simGetObjectPosition(targetHandle,-1)
                    billP=simGetObjectPosition(BillHandle,-1)
                    vv={targetP[1]-billP[1],targetP[2]-billP[2]}
                    if (math.sqrt(vv[1]*vv[1]+vv[2]*vv[2])>0.2) then
                        pathCalculated=0 -- We have to recompute the path
                    end
                end
            end
        end
    
    
    
    vel=(rightV+leftV)*0.5*0.8/0.56
    if (vel<0) then vel=0 end
    
    scaling=(vel/nominalVelocity)/1.4
    
    vp=vp+simGetSimulationTimeStep()*vel
    p=math.fmod(vp,1)
    indexLow=math.floor(p/dl)
    t=p/dl-indexLow
    oppIndexLow=math.floor(indexLow+tl/2)
    if (oppIndexLow>=tl) then oppIndexLow=oppIndexLow-tl end
    indexHigh=indexLow+1
    if (indexHigh>=tl) then indexHigh=indexHigh-tl end
    oppIndexHigh=oppIndexLow+1
    if (oppIndexHigh>=tl) then oppIndexHigh=oppIndexHigh-tl end
    
    simSetJointPosition(legJointHandles[1],(legWaypoints[indexLow+1]*(1-t)+legWaypoints[indexHigh+1]*t)*scaling)
    simSetJointPosition(kneeJointHandles[1],(kneeWaypoints[indexLow+1]*(1-t)+kneeWaypoints[indexHigh+1]*t)*scaling)
    simSetJointPosition(ankleJointHandles[1],(ankleWaypoints[indexLow+1]*(1-t)+ankleWaypoints[indexHigh+1]*t)*scaling)
    simSetJointPosition(shoulderJointHandles[1],(shoulderWaypoints[indexLow+1]*(1-t)+shoulderWaypoints[indexHigh+1]*t)*scaling)
    simSetJointPosition(elbowJointHandles[1],(elbowWaypoints[indexLow+1]*(1-t)+elbowWaypoints[indexHigh+1]*t)*scaling)
    
    simSetJointPosition(legJointHandles[2],(legWaypoints[oppIndexLow+1]*(1-t)+legWaypoints[oppIndexHigh+1]*t)*scaling)
    simSetJointPosition(kneeJointHandles[2],(kneeWaypoints[oppIndexLow+1]*(1-t)+kneeWaypoints[oppIndexHigh+1]*t)*scaling)
    simSetJointPosition(ankleJointHandles[2],(ankleWaypoints[oppIndexLow+1]*(1-t)+ankleWaypoints[oppIndexHigh+1]*t)*scaling)
    simSetJointPosition(shoulderJointHandles[2],(shoulderWaypoints[oppIndexLow+1]*(1-t)+shoulderWaypoints[oppIndexHigh+1]*t)*scaling)
    simSetJointPosition(elbowJointHandles[2],(elbowWaypoints[oppIndexLow+1]*(1-t)+elbowWaypoints[oppIndexHigh+1]*t)*scaling)
    
    linMov=s*simGetSimulationTimeStep()*(rightV+leftV)*0.5*scaling*(relativeVel[indexLow+1]*(1-t)+relativeVel[indexHigh+1]*t)
    rotMov=simGetSimulationTimeStep()*math.atan((rightV-leftV)*8)
    position=simGetObjectPosition(BillHandle,sim_handle_parent)
    orientation=simGetObjectOrientation(BillHandle,sim_handle_parent)
    xDir={math.cos(orientation[3]),math.sin(orientation[3]),0.0}
    position[1]=position[1]+xDir[1]*linMov
    position[2]=position[2]+xDir[2]*linMov
    orientation[3]=orientation[3]+rotMov
    simSetObjectPosition(BillHandle,sim_handle_parent,position)
    simSetObjectOrientation(BillHandle,sim_handle_parent,orientation)
end 
