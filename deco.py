from tqdm import tqdm
import time
from colorama import Fore, Style, init
init(autoreset=True)

class ProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["ascii"] = " ⠋⠙⠸⠴⠦⠇#"
        #kwargs["ascii"] = False
        kwargs["bar_format"] = (
            f"{Fore.CYAN}[{{desc}}]{Style.RESET_ALL} "
            f"{Fore.YELLOW}{{percentage:3.0f}}%{Style.RESET_ALL} "
            f"{Fore.GREEN}{{bar}}{Style.RESET_ALL} "
            "[{n_fmt}/{total_fmt}] ⏱️ {elapsed}<{remaining} {postfix}"
        )
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    # 예시 사용
    for i in ProgressBar(range(100), desc="In progress", ncols=80):
        time.sleep(0.05)
