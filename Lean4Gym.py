import subprocess
import os
import time
import select
import json

_REPL_PROMPT = "REPL>"

class ProofState:
    def __init__(self, result):
        res = json.loads(result)
        self.sid = res["sid"]
        self.tacticState = res["tacticState"]
        self.error = res["error"]
        if self.tacticState == "no goals":
            self.finishFlag = True
        else:
            self.finishFlag = False

    def isFinish(self):
        return self.finishFlag

    def getTacticState(self):
        return self.tacticState

    def getStateID(self):
        return self.sid

    def getError(self):
        return self.error

    def print(self):
        print("sid:", self.sid)
        print("tacticState:", self.tacticState)
        print("error:", self.error)


class Lean4Gym:
    def __init__(self, lean_workdir, lean_file):
        self.proc = subprocess.Popen(["lake", "env", "lean", lean_file],
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     text=True,
                                     universal_newlines=True,
                                     bufsize=1,
                                     cwd=lean_workdir)
        time.sleep(0.05)
        # self.initState = ProofState(self.__readResult__())

    def getInitState(self):
        # return self.initState
        return ProofState(self.__readResult__())

    def run_tactic(self, state, tactic):
        command = {
            "sid": state.getStateID(),
            "cmd": tactic
        }
        str_command = json.dumps(command)
        str_command = "{}{}".format(str_command, "\n")
        # print("str_command:", str_command)
        # print("---")

        return ProofState(self.__cmd__(str_command))

    def __readResult__(self):
        if self.proc.stdout is None:
            raise RuntimeError("self.proc.stout is not initialized")

        while True:
            line = self.proc.stdout.readline().strip()
            # logger.debug(line)
            if line == "":
                print("Line is empty!")
                raise EOFError
            if line.startswith(_REPL_PROMPT):
                return line[len(_REPL_PROMPT):].strip()
        return ""

    def __cmd__(self, command):
        self.proc.stdin.write(command)
        self.proc.stdin.flush()
        return self.__readResult__()

    def close(self):
        self.proc.stdin.close()
        self.proc.kill()  # 终止进程




