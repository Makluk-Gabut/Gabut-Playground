import pygame
import random
import json
import os
import sys

pygame.init()

# ==========================================
# AUDIO
# ==========================================
try:
    pygame.mixer.init()
    AUDIO_READY = True
except Exception:
    AUDIO_READY = False

# ==========================================
# CONFIG CLASS
# ==========================================
class CFG:

    WIDTH = 900
    HEIGHT = 600

    SNAKE_SIZE = 15

    START_SPEED = 10
    MAX_SPEED = 35

    FPS_MENU = 15
    FPS_PAUSE = 15
    FPS_GAME_OVER = 15

    HIGH_SCORE_FILE = "highscore.json"

    COLORS = {
        "white": (255, 255, 255),
        "yellow": (255, 255, 102),
        "background": (30, 30, 30),
        "red": (213, 50, 80),
        "green": (0, 255, 100),
        "head": (0, 200, 0),
        "border": (70, 70, 70),
        "level": (0, 220, 255),
        "menu": (50, 153, 213)
    }

# ==========================================
# DISPLAY
# ==========================================
screen = pygame.display.set_mode(
    (CFG.WIDTH, CFG.HEIGHT)
)

pygame.display.set_caption(
    "Game Uler-ulean "
)

icon = pygame.Surface((32, 32))
icon.fill(CFG.COLORS["green"])

pygame.display.set_icon(icon)

clock = pygame.time.Clock()

# ==========================================
# FONT
# ==========================================
score_font = pygame.font.SysFont(
    "consolas",
    25
)

message_font = pygame.font.SysFont(
    "arial",
    40,
    bold=True
)

sub_font = pygame.font.SysFont(
    "arial",
    20
)

small_font = pygame.font.SysFont(
    "arial",
    24,
    bold=True
)

# ==========================================
# SOUND
# ==========================================
eat_sound = None
crash_sound = None

if AUDIO_READY:

    try:
        eat_sound = pygame.mixer.Sound(
            "makan.wav"
        )
    except Exception:
        pass

    try:
        crash_sound = pygame.mixer.Sound(
            "kalah.wav"
        )
    except Exception:
        pass

# ==========================================
# DISPLAY MODE
# ==========================================
def toggle_fullscreen():

    global screen

    fullscreen = bool(
        pygame.display.get_surface().get_flags()
        & pygame.FULLSCREEN
    )

    if fullscreen:

        screen = pygame.display.set_mode(
            (CFG.WIDTH, CFG.HEIGHT)
        )

    else:

        screen = pygame.display.set_mode(
            (CFG.WIDTH, CFG.HEIGHT),
            pygame.FULLSCREEN
        )

    pygame.display.set_caption(
        "Snake Game Pro 2026"
    )

# ==========================================
# HIGH SCORE
# ==========================================
def load_high_score():

    if not os.path.exists(
        CFG.HIGH_SCORE_FILE
    ):
        return 0

    try:

        with open(
            CFG.HIGH_SCORE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return int(
            data.get("highscore", 0)
        )

    except Exception:
        return 0


def save_high_score(value):

    try:

        with open(
            CFG.HIGH_SCORE_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                {"highscore": int(value)},
                file
            )

    except Exception:
        pass

# ==========================================
# DRAW UI
# ==========================================
def draw_score(
    score,
    level,
    high_score
):

    score_text = score_font.render(
        f"Skor: {score}",
        True,
        CFG.COLORS["white"]
    )

    level_text = score_font.render(
        f"Level: {level}",
        True,
        CFG.COLORS["yellow"]
    )

    high_text = score_font.render(
        f"High Score: {high_score}",
        True,
        CFG.COLORS["menu"]
    )

    screen.blit(score_text, (10, 10))
    screen.blit(level_text, (10, 40))

    screen.blit(
        high_text,
        (CFG.WIDTH - 250, 10)
    )


def center_text(
    title,
    subtitle="",
    color=(255, 255, 255),
    y_shift=0
):

    title_surface = message_font.render(
        title,
        True,
        color
    )

    title_rect = title_surface.get_rect(
        center=(
            CFG.WIDTH // 2,
            CFG.HEIGHT // 2 - 20 + y_shift
        )
    )

    screen.blit(
        title_surface,
        title_rect
    )

    if subtitle:

        sub_surface = sub_font.render(
            subtitle,
            True,
            CFG.COLORS["white"]
        )

        sub_rect = sub_surface.get_rect(
            center=(
                CFG.WIDTH // 2,
                CFG.HEIGHT // 2 + 30 + y_shift
            )
        )

        screen.blit(
            sub_surface,
            sub_rect
        )

# ==========================================
# DRAW SNAKE
# ==========================================
def draw_snake(snake_list):

    for index, segment in enumerate(snake_list):

        color = (
            CFG.COLORS["head"]
            if index == len(snake_list) - 1
            else CFG.COLORS["green"]
        )

        pygame.draw.rect(
            screen,
            color,
            (
                segment[0],
                segment[1],
                CFG.SNAKE_SIZE,
                CFG.SNAKE_SIZE
            )
        )

        pygame.draw.rect(
            screen,
            CFG.COLORS["border"],
            (
                segment[0],
                segment[1],
                CFG.SNAKE_SIZE,
                CFG.SNAKE_SIZE
            ),
            1
        )

# ==========================================
# DRAW FOOD
# ==========================================
def draw_food(food_pos):

    if food_pos is None:
        return

    fx, fy = food_pos

    pygame.draw.rect(
        screen,
        CFG.COLORS["red"],
        (
            fx,
            fy,
            CFG.SNAKE_SIZE,
            CFG.SNAKE_SIZE
        ),
        border_radius=4
    )

    pygame.draw.rect(
        screen,
        CFG.COLORS["border"],
        (
            fx,
            fy,
            CFG.SNAKE_SIZE,
            CFG.SNAKE_SIZE
        ),
        1
    )

# ==========================================
# FOOD SPAWN
# ==========================================
def spawn_food(snake_list):

    total_x = (
        CFG.WIDTH //
        CFG.SNAKE_SIZE
    )

    total_y = (
        CFG.HEIGHT //
        CFG.SNAKE_SIZE
    )

    total_tiles = total_x * total_y

    if len(snake_list) >= total_tiles:
        return None

    occupied = {
        tuple(part)
        for part in snake_list
    }

    empty_positions = [

        (
            x * CFG.SNAKE_SIZE,
            y * CFG.SNAKE_SIZE
        )

        for x in range(total_x)
        for y in range(total_y)

        if (
            x * CFG.SNAKE_SIZE,
            y * CFG.SNAKE_SIZE
        ) not in occupied
    ]

    if not empty_positions:
        return None

    return random.choice(
        empty_positions
    )

# ==========================================
# MAIN MENU
# ==========================================
def show_main_menu(high_score):

    while True:

        screen.fill(
            CFG.COLORS["background"]
        )

        center_text(
            "SNAKE GAME PRO 2026",
            "SPACE: Mulai | F: Fullscreen | Q: Keluar",
            CFG.COLORS["menu"],
            y_shift=-40
        )

        hs_surface = small_font.render(
            f"High Score: {high_score}",
            True,
            CFG.COLORS["white"]
        )

        screen.blit(
            hs_surface,
            hs_surface.get_rect(
                center=(
                    CFG.WIDTH // 2,
                    CFG.HEIGHT // 2 + 55
                )
            )
        )

        pygame.display.update()

        clock.tick(
            CFG.FPS_MENU
        )

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_q:
                    return False

                if event.key == pygame.K_SPACE:
                    return True

                if event.key == pygame.K_f:
                    toggle_fullscreen()

# ==========================================
# PAUSE
# ==========================================
def show_pause_screen(
    score,
    level,
    high_score
):

    while True:

        screen.fill(
            CFG.COLORS["background"]
        )

        center_text(
            "PAUSED",
            "P/SPACE: Lanjut | Q: Keluar | F: Fullscreen",
            CFG.COLORS["yellow"],
            y_shift=-30
        )

        draw_score(
            score,
            level,
            high_score
        )

        pygame.display.update()

        clock.tick(
            CFG.FPS_PAUSE
        )

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:

                if event.key in (
                    pygame.K_p,
                    pygame.K_SPACE
                ):
                    return True

                if event.key == pygame.K_q:
                    return False

                if event.key == pygame.K_f:
                    toggle_fullscreen()

# ==========================================
# END SCREEN
# ==========================================
def show_end_screen(
    title,
    subtitle,
    color,
    score,
    level,
    high_score
):

    while True:

        screen.fill(
            CFG.COLORS["background"]
        )

        center_text(
            title,
            subtitle,
            color,
            y_shift=-10
        )

        draw_score(
            score,
            level,
            high_score
        )

        pygame.display.update()

        clock.tick(
            CFG.FPS_GAME_OVER
        )

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_q:
                    return False

                if event.key == pygame.K_c:
                    return True

                if event.key == pygame.K_f:
                    toggle_fullscreen()

# ==========================================
# GAME LOOP
# ==========================================
def game_loop():

    high_score = load_high_score()

    x = (
        CFG.WIDTH // 2 //
        CFG.SNAKE_SIZE
    ) * CFG.SNAKE_SIZE

    y = (
        CFG.HEIGHT // 2 //
        CFG.SNAKE_SIZE
    ) * CFG.SNAKE_SIZE

    # langsung jalan
    velocity_x = CFG.SNAKE_SIZE
    velocity_y = 0

    snake_list = []
    snake_length = 1

    score = 0
    level = 1

    speed = CFG.START_SPEED

    level_up_timer = 0

    food_pos = spawn_food(
        snake_list
    )

    if food_pos is None:
        return False

    while True:

        direction_changed = False

        # ==================================
        # INPUT
        # ==================================
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_f:
                    toggle_fullscreen()
                    continue

                if event.key in (
                    pygame.K_p,
                    pygame.K_SPACE
                ):

                    if not show_pause_screen(
                        score,
                        level,
                        high_score
                    ):
                        return False

                    continue

                # Anti input buffering
                if not direction_changed:

                    if (
                        event.key == pygame.K_LEFT
                        and velocity_x != CFG.SNAKE_SIZE
                    ):

                        velocity_x = -CFG.SNAKE_SIZE
                        velocity_y = 0

                        direction_changed = True

                    elif (
                        event.key == pygame.K_RIGHT
                        and velocity_x != -CFG.SNAKE_SIZE
                    ):

                        velocity_x = CFG.SNAKE_SIZE
                        velocity_y = 0

                        direction_changed = True

                    elif (
                        event.key == pygame.K_UP
                        and velocity_y != CFG.SNAKE_SIZE
                    ):

                        velocity_y = -CFG.SNAKE_SIZE
                        velocity_x = 0

                        direction_changed = True

                    elif (
                        event.key == pygame.K_DOWN
                        and velocity_y != -CFG.SNAKE_SIZE
                    ):

                        velocity_y = CFG.SNAKE_SIZE
                        velocity_x = 0

                        direction_changed = True

        # ==================================
        # UPDATE POSITION
        # ==================================
        x += velocity_x
        y += velocity_y

        # ==================================
        # WALL COLLISION
        # ==================================
        if (

            x < 0

            or x + CFG.SNAKE_SIZE > CFG.WIDTH

            or y < 0

            or y + CFG.SNAKE_SIZE > CFG.HEIGHT
        ):

            if crash_sound:
                crash_sound.play()

            if not show_end_screen(
                "GAME OVER!",
                "C: Main Lagi | Q: Keluar",
                CFG.COLORS["red"],
                score,
                level,
                high_score
            ):
                return False

            return True

        # ==================================
        # RENDER
        # ==================================
        screen.fill(
            CFG.COLORS["background"]
        )

        draw_food(food_pos)

        snake_head = [x, y]

        snake_list.append(
            snake_head
        )

        if len(snake_list) > snake_length:
            del snake_list[0]

        # Self collision
        if snake_head in snake_list[:-1]:

            if crash_sound:
                crash_sound.play()

            if not show_end_screen(
                "GAME OVER!",
                "C: Main Lagi | Q: Keluar",
                CFG.COLORS["red"],
                score,
                level,
                high_score
            ):
                return False

            return True

        draw_snake(
            snake_list
        )

        # ==================================
        # LEVEL UP EFFECT
        # ==================================
        if level_up_timer > 0:

            level_surface = message_font.render(
                f"LEVEL {level}!",
                True,
                CFG.COLORS["level"]
            )

            level_rect = level_surface.get_rect(
                center=(
                    CFG.WIDTH // 2,
                    70
                )
            )

            screen.blit(
                level_surface,
                level_rect
            )

            level_up_timer -= 1

        draw_score(
            score,
            level,
            high_score
        )

        pygame.display.update()

        # ==================================
        # EAT FOOD
        # ==================================
        if (
            food_pos is not None
            and snake_head == list(food_pos)
        ):

            if eat_sound:
                eat_sound.play()

            snake_length += 1

            score += 1

            if score > high_score:

                high_score = score

                save_high_score(
                    high_score
                )

            new_level = (
                score // 5
            ) + 1

            if new_level > level:

                level = new_level

                level_up_timer = 30

            speed = min(
                CFG.START_SPEED + (level * 2),
                CFG.MAX_SPEED
            )

            food_pos = spawn_food(
                snake_list
            )

            # WIN CONDITION
            if food_pos is None:

                if not show_end_screen(
                    "KAMU MENANG!",
                    "C: Main Lagi | Q: Keluar",
                    CFG.COLORS["level"],
                    score,
                    level,
                    high_score
                ):
                    return False

                return True

        clock.tick(speed)

# ==========================================
# MAIN
# ==========================================
def main():

    high_score = load_high_score()

    if not show_main_menu(
        high_score
    ):

        pygame.quit()
        sys.exit()

    running = True

    while running:
        running = game_loop()

    pygame.quit()
    sys.exit()

# ==========================================
# START
# ==========================================
if __name__ == "__main__":
    main()
