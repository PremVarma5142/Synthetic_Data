import random
import cv2
import numpy as np 

class transparent():

    def __init__(self, center=False, num_overlays=1):

        self.center = center
        self.num_overlays = num_overlays
       
    def adjust_coordinates(self, x, y, box_width, box_height, overlay_w, overlay_h, bg_w, bg_h, bg_x, bg_y):
        x = (x * overlay_w + bg_x) / bg_w
        y = (y * overlay_h + bg_y) / bg_h
        box_width = box_width * overlay_w / bg_w
        box_height = box_height * overlay_h / bg_h
        return x, y, box_width, box_height
    
    def add_transparent_image(self, background, foreground, lines, x_offset=None, y_offset=None, flip=True, rotate=True):
        bg_h, bg_w, bg_channels = background.shape
        fg_h, fg_w, fg_channels = foreground.shape
        if fg_w > bg_w or fg_h > bg_h:
            return []

        assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
        assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

        if self.center and self.num_overlays > 1:
            raise ValueError("center=True is applicable only when num_overlays is 1.")

        if self.center:
            x_offset = (bg_w - fg_w) // 2
            y_offset = (bg_h - fg_h) // 2
        else:
            if x_offset is None:
                x_offset = random.randint(100, bg_w - fg_w - 100)
            if y_offset is None:
                y_offset = random.randint(100, bg_h - fg_h - 100)

        if flip and random.random() < 0.5:
            foreground = cv2.flip(foreground, 1)  # 1 for horizontal flip
            x_offset = bg_w - x_offset - fg_w  # Adjust x-coordinate after flipping

            for i in range(len(lines)):
                line = lines[i].split(" ")
                index = int(line[0])
                x, y, box_width, box_height = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                x = 1.0 - x 
                lines[i] = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(index, x, y, box_width, box_height)

        if rotate:
            rotation_angle = 0  
            if random.random() < 0.25:
                rotation_angle = 90  
            elif random.random() < 0.25:
                rotation_angle = 180 
            elif random.random() < 0.25:
                rotation_angle = 270
            
            if rotation_angle == 90:
                foreground = cv2.rotate(foreground, cv2.ROTATE_90_CLOCKWISE)
                fg_w, fg_h = fg_h, fg_w

                for i in range(len(lines)):
                    line = lines[i].split(" ")
                    index = int(line[0])
                    x, y, box_width, box_height = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                    new_x = (1.0 - (y + box_height / 2)) + box_height / 2
                    new_y = x + box_width / 2 - box_width / 2
                    new_width = box_height
                    new_height = box_width
                    lines[i] = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(index, new_x, new_y, new_width, new_height)

                if x_offset is not None and y_offset is not None:
                    new_x_offset = x_offset - (fg_h - fg_w) // 2
                    new_y_offset = y_offset + (fg_h - fg_w) // 2
                    x_offset, y_offset = new_x_offset, new_y_offset

            elif rotation_angle == 180:
                foreground = cv2.rotate(foreground, cv2.ROTATE_180)
                fg_w, fg_h = fg_w, fg_h  

                for i in range(len(lines)):
                    line = lines[i].split(" ")
                    index = int(line[0])
                    x, y, box_width, box_height = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                    new_x = 1.0 - x
                    new_y = 1.0 - y
                    new_width = box_width
                    new_height = box_height
                    lines[i] = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(index, new_x, new_y, new_width, new_height)

                if x_offset is not None and y_offset is not None:
                    new_x_offset = x_offset
                    new_y_offset = y_offset
                    x_offset, y_offset = new_x_offset, new_y_offset

            elif rotation_angle == 270:
                foreground = cv2.rotate(foreground, cv2.ROTATE_90_CLOCKWISE)
                fg_w, fg_h = fg_h, fg_w

                for i in range(len(lines)):
                    line = lines[i].split(" ")
                    index = int(line[0])
                    x, y, box_width, box_height = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                    new_x = (1.0 - (y + box_height / 2)) + box_height / 2
                    new_y = x + box_width / 2 - box_width / 2
                    new_width = box_height
                    new_height = box_width
                    lines[i] = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(index, new_x, new_y, new_width, new_height)

                if x_offset is not None and y_offset is not None:
                    new_x_offset = x_offset - (fg_h - fg_w) // 2
                    new_y_offset = y_offset + (fg_h - fg_w) // 2
                    x_offset, y_offset = new_x_offset, new_y_offset

        w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
        h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

        if w < 1 or h < 1:
            return[]

        bg_x = max(0, x_offset)
        bg_y = max(0, y_offset)
        fg_x = max(0, x_offset * -1)
        fg_y = max(0, y_offset * -1)
        foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
        background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]
        foreground_colors = foreground[:, :, :3]
        alpha_channel = foreground[:, :, 3] / 255
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
        composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask
        background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
        new_lines = []
        for line in lines:
            line = line.split(" ")
            index = int(line[0])
            x, y, box_width, box_height = float(line[1]), float(line[2]), float(line[3]), float(line[4])
            x, y, box_width, box_height = self.adjust_coordinates(x, y, box_width, box_height, fg_w, fg_h, bg_w, bg_h, bg_x, bg_y)
            new_lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(index, x, y, box_width, box_height))

        return new_lines
