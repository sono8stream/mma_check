/*
 * Copyright (c) 2018-2019, Texas Instruments Incorporated
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * *  Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * *  Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * *  Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//
//File Name: launch_j721e.js
//Description:
//   Launch the DMSC firmware and board configuration from R5F.
//
//Usage:
//
//From CCS Scripting console
//  1. loadJSFile "C:\\ti\\launch_j721e.js"
//
//Note:
//  1. Search for "edit this" to look at changes that need to be edited
//     for your usage.
//

//<!!!!!! EDIT THIS !!!!!>
// Set this to 1, if using 'FreeRTOS'
isFreertos = 1;
// Set this to 1, if using 'SafeRTOS'
isSafertos = 0;

//<!!!!!! EDIT THIS !!!!!>

// Import the DSS packages into our namespace to save on typing
importPackage(Packages.com.ti.debug.engine.scripting);
importPackage(Packages.com.ti.ccstudio.scripting.environment);
importPackage(Packages.java.lang);
importPackage(Packages.java.io);

function updateScriptVars() {
  //Open a debug session
  dsDMSC_0 = debugServer.openSession(".*DMSC_Cortex_M3_0");
}

function printVars() {
  updateScriptVars();
}

function connectTargets() {
  /* Set timeout of 20 seconds */
  script.setScriptTimeout(200000);
  updateScriptVars();
  sysResetVar = dsDMSC_0.target.getResetType(1);
  sysResetVar.issueReset();
  print("Connecting to DMSC_Cortex_M3_0!");
  // Connect targets
  dsDMSC_0.target.connect();
  // Run the DDR Configuration
  print("Running the DDR configuration... Wait till it completes!");
  dsDMSC_0.target.halt();
  dsDMSC_0.expression.evaluate("J7ES_LPDDR4_Config()");
  dsDMSC_0.target.runAsynch();

  print("Connecting to MCU Cortex_R5_0!");
}

function disconnectTargets() {
  updateScriptVars();
  // Disconnect targets
  dsDMSC_0.target.disconnect();
}

function loadC7x() {
  dsC7X1_0 = debugServer.openSession(".*C71X_0");

  dsC7X1_0.target.connect();

  c7x_binary =
    "C:\\Users\\sono8\\workspace_v12\\HelloC7\\DebugWithAsm\\HelloC7.out";

  dsC7X1_0.memory.loadProgram(c7x_binary);
}

function doEverything() {
  printVars();
  connectTargets();
  disconnectTargets();
  loadC7x();
  print("Happy Debugging!!");
}

var ds;
var debugServer;
var script;

// Check to see if running from within CCSv4 Scripting Console
var withinCCS = ds !== undefined;

// Create scripting environment and get debug server if running standalone
if (!withinCCS) {
  // Import the DSS packages into our namespace to save on typing
  importPackage(Packages.com.ti.debug.engine.scripting);
  importPackage(Packages.com.ti.ccstudio.scripting.environment);
  importPackage(Packages.java.lang);

  // Create our scripting environment object - which is the main entry point into any script and
  // the factory for creating other Scriptable ervers and Sessions
  script = ScriptingEnvironment.instance();

  // Get the Debug Server and start a Debug Session
  debugServer = script.getServer("DebugServer.1");
} // otherwise leverage existing scripting environment and debug server
else {
  debugServer = ds;
  script = env;
}

doEverything();
