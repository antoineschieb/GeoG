from multiprocessing import Process
from sample import task
import time


def start_new_process(processes):
    p = Process(target=task)
    p.start()
    processes.append(p)
    return


if __name__ == '__main__':
    # create the processes
    processes = []
    for i in range(4):
        time.sleep(15)
        start_new_process(processes)

    total_killed = 0
    while True:
        time.sleep(30)
        print(f"[distribute.py] ==================================================")
        for p in processes:
            # report the process is alive
            if not p.is_alive():
                print(f"[distribute.py] : process {p} died (or finished), spawning new one.")
                processes.remove(p)
                total_killed += 1
                start_new_process(processes)

        print(f"[distribute.py] : {len(processes)} running (killed {total_killed} total)")

