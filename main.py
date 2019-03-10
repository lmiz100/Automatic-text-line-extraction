
from tkinter import *
from PIL import Image
import cv2
import numpy as np
import os


global img, err, flat, is_text_row, first_text_row, last_text_row, res, contours, precision, nw_first_row, nw_last_row, nw_index, nw_space_avg

#err is a flag, value 1 when something went wrong
err = 1
img = None
flat = []


def quit_program():
    exit()


def set_error(value, msg):
    global err
    err = value
    output_label["text"] = msg

    return


def find_next_line_row(points_sortByRow,last_line):
    for point in points_sortByRow:
        if point[1] > last_line:
            return point[1]

    return -1


def prepare_image():
    global img, res, flat, is_text_row, first_text_row, contours

    blur = img
    # apply Gaussian filter
    blur = cv2.GaussianBlur(blur, (5, 5), 0)

    # threshold
    global threshed
    th, threshed = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    #show_image("blur", blur)
    #show_image("thresh", threshed)

    #get all text points
    pts = cv2.findNonZero(threshed)

    flat = []
    # after the loop, flat is sorted by row, when row is the right number at each array
    for sublist in pts:
        for item in sublist:
            flat.append(item)

    row_num, col_num = threshed.shape

    # initial new dictionary
    is_text_row = {x: x * 0 for x in range(0, row_num)}
    for point in flat:
        is_text_row[point[1]] = 1

    first_text_row = find_next_line_row(flat, 0)  # first text row

    #find contours and reverse the array for chronological row order
    _, contours, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[::-1]

    res = img

    return


def enter_image():
    name = input_entry.get()
    if name == "":
        set_error(1, "Please enter image name")
    else:
        # read image in grayscale
        global img, precision, last_text_row, nw_index, nw_first_row
        #img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        path = os.path.join(os.path.dirname(__file__), 'test images')
        img = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            set_error(1, "Illegal image name, try again")
        else:
            set_error(0, "Reading success!!")
            input_entry.delete(0, 'end')
            precision = 1
            last_text_row = -1
            nw_index = -1
            nw_first_row = 0
            prepare_image()


    return


def find_next_blank_row(img, curr_row):
    row_num, col_num = img.shape

    if curr_row >= row_num:  # Illegal row number
        return -1

    for x in range(curr_row, row_num):
        if is_text_row[x] == 0:
            return x

    return -1


def show_image(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_line_contour():
    line_contour = None
    for x in contours:
        if first_text_row <= x[0][0][1] <= last_text_row:
            if line_contour is None:
                line_contour = x
            else:
                line_contour = np.concatenate((line_contour, x))
        elif x[0][0][1] > last_text_row:
            break

    return line_contour


def find_line():
    global last_text_row, first_text_row, res, precision, nw_first_row, nw_index
    output_label["text"] = ""

    if err == 1:
        set_error(err, "Please enter image before")
    elif first_text_row == -1:
        set_error(0, "No more lines in image")
    else:
        #path = 'C:/Users/lior_/PycharmProjects/final text line extraction/temp'
        path = os.path.join(os.path.dirname(__file__), 'temp')
        if nw_index != -1:
            res = cv2.imread(os.path.join(path, "before_nw.png"), cv2.IMREAD_GRAYSCALE)
        precision = 1
        if last_text_row != -1:
            first_text_row = find_next_line_row(flat, last_text_row)

        if first_text_row != -1:
            # temp_image doesn't save the last line - for improve accuracy purpose
            try:
                cv2.imwrite(os.path.join(path, "temp_image.png"), res)
            except RuntimeError:
                set_error(0, "Can't save, Try other name")
                return

            last_text_row = find_next_blank_row(threshed, first_text_row)
            hull = [cv2.convexHull(get_line_contour())]
            cv2.drawContours(res, hull, -1, (100, 100, 100), 1)
            show_image("Find line result", res)

            nw_first_row = -2
            nw_index = -1
        else:
            set_error(0, "No more lines in image")

    return


# save as .png image for default
def save_image():
    name = input_entry.get()

    if err == 1:
        set_error(err, "No image to save")
        return
    elif name == "":
        set_error(0, "Please enter new image name")
        return
    else:
        try:
            #path = 'C:/Users/lior_/PycharmProjects/final text line extraction/results'
            path = os.path.join(os.path.dirname(__file__), 'results')
            cv2.imwrite(os.path.join(path, name + ".png"), res)
        except RuntimeError:
            set_error(0, "Can't save, Try other name")
            return

    set_error(0, "Image saved as " + name+".png")
    input_entry.delete(0, 'end')

    return


#find how many arrays compose line contour
def get_np_length():
    acc = 0
    for x in contours:
        if first_text_row <= x[0][0][1] <= last_text_row:
            acc += 1
        elif x[0][0][1] > last_text_row:
            break

    return acc


def get_row_edges(row):
    left = -1
    right = -1

    for point in flat:
        if point[1] == row and left == -1:
            left = point[0]
        elif point[1] == row:
            right = point[0]
        elif point[1] > row:
            return [left, right]

    return [left, right]


def find_col_text_range(first_row, last_row):
    most_left = 1000000
    most_right = -1

    if first_row == -1 or last_row == -1:
        return [-1, -1]

    for row in range(first_row, last_row):
        edges = get_row_edges(row)
        if most_left > edges[0]:
            most_left = edges[0]
        if most_right < edges[1]:
            most_right = edges[1]

    return [most_left, most_right]


def get_partial_line_contour(first_row, last_row, col_range):
    partial_line_contour = None

    for x in contours:
        if first_row <= x[0][0][1] <= last_row:
            if col_range[0] <= x[0][0][0] <= col_range[1]:
                if partial_line_contour is None:
                    partial_line_contour = x
                else:
                    partial_line_contour = np.concatenate((partial_line_contour, x))
        elif x[0][0][1] > last_row:
            break

    return partial_line_contour


#side is "right" if the contour is from the left contour, and "left" for right contour
def find_linking_points(side, initial_value, contour):
    col = initial_value
    upper_limit = 1000000
    bottom_limit = 0
    high_row = -1
    low_row = 1000000
    middle_row = round((last_text_row - first_text_row) / 2) + first_text_row

    if contour is None:
        return

    #print("side:", side, "init:",initial_value, "cont:", contour)
    while True:
        if side == "right":
            for point in contour:
                if col < point[0][0] <= upper_limit:
                    col = point[0][0]
        elif side == "left":
            for point in contour:
                if bottom_limit <= point[0][0] < col:
                    col = point[0][0]

        if low_row > middle_row:
            for point in contour:
                #print("midle:", middle_row, "point", point, "col:", col)
                if point[0][0] == col:
                    if point[0][1] < low_row and point[0][1] < middle_row:
                        low_row = point[0][1]
        if high_row < middle_row:
            for point in contour:
                if point[0][0] == col:
                    if point[0][1] > high_row and point[0][1] > middle_row:
                        high_row = point[0][1]
        #print("low", low_row,"high", high_row)
        if low_row == high_row or low_row > middle_row or high_row < middle_row:
            if side == "right":
                upper_limit = col - 1
                col = -1
            else:
                bottom_limit = col + 1
                col = 1000000
        else:
            break

    return [(col, low_row), (col, high_row)]


#devide line contour to precision*2 parts, and add convex hull for each part
def devide_line_contour():
    col_range = find_col_text_range(first_text_row, last_text_row)
    #print("col range", col_range)

    row_length = col_range[1] - col_range[0] +1
    part_width = round(row_length / pow(2, precision))

    low_col = col_range[0]
    high_col = low_col + part_width -1
    counter = 0
    upper_point = -1
    lower_point = -1
    prev_right = None
    while counter < pow(2, precision):
        partial = get_partial_line_contour(first_text_row, last_text_row, [low_col, high_col])

        if partial is not None:
            add_convexHull(res, partial)

            if prev_right is None:
                prev_right = find_linking_points("right", -1, partial)
            else:
                linking_points = find_linking_points("left", 1000000, partial)
                #don't draw line if the parts are overlap
                if prev_right[0][0] < linking_points[0][0] and prev_right[1][0] < linking_points[1][0]:
                    cv2.line(res, prev_right[0], linking_points[0], (0, 255, 0), 1)
                    cv2.line(res, prev_right[1], linking_points[1], (0, 255, 0), 1)
                prev_right = find_linking_points("right", -1, partial)

        low_col = high_col + 1
        high_col += part_width
        if high_col > col_range[1]:
            high_col = col_range[1]
        counter += 1

    show_image("Dirty improved result", res)

    return


def add_convexHull(image, cont):
    hull = [cv2.convexHull(cont)]
    cv2.drawContours(image, hull, -1, (100, 100, 100), 1)
    #print("cont ", cont)
    #print("hull", hull)

    return


#return the number of the line contour (last contour in row range. working with np array is hard)
def get_improve_contour_index(contour):
    i = 0
    for cont in contour:
        if cont[0][0][1] > last_text_row:
            break
        i += 1

    return i - 1


def improve_accuracy():
    global precision, res
    output_label["text"] = ""

    if err == 1:
        set_error(err, "Please enter image before")
    elif last_text_row == -1:
        set_error(0, "Please find line before")
    elif first_text_row == -1:
        set_error(0, "No more lines in image")
    else:
        # temp_image doesn't save the last line - for improve accuracy purpose
        #path = 'C:/Users/lior_/PycharmProjects/final text line extraction/temp'
        path = os.path.join(os.path.dirname(__file__), 'temp')
        if precision == 1:
            cv2.imwrite(os.path.join(path, "before_Improved.png"), res)
        res = cv2.imread(os.path.join(path, "temp_image.png"), cv2.IMREAD_GRAYSCALE)
        devide_line_contour()
        precision += 1
        cv2.imwrite(os.path.join(path, "Improved.png"), res)

        #now we have "dirty" and more accurate draw in Improved.png. This code make it cleaner
        improved_img = cv2.imread(os.path.join(path, "Improved.png"), cv2.IMREAD_GRAYSCALE)
        _, new_threshed = cv2.threshold(improved_img, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # find contours and reverse the array for chronological row order
        _, improved_contours, _ = cv2.findContours(new_threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        improved_contours = improved_contours[::-1]

        res = cv2.imread(os.path.join(path, "temp_image.png"), cv2.IMREAD_GRAYSCALE)
        cv2.drawContours(res, improved_contours, get_improve_contour_index(improved_contours), (100, 100, 100), 1)
        cv2.imwrite(os.path.join(path, "Improved.png"), res)
        show_image("Clean improved result", res)

        return


#when text is 1, return next text col, else return next space col
def get_next_col(text, col_range):
    space_sum = 0
    #define the sensitivity level
    row_height = nw_last_row - nw_first_row
    avg_multiplier = row_height / (50 * (1 + row_height // 50))

    if text == 1:
        for x in range(nw_index[1] + 1, col_range[1]):
            for y in range(nw_first_row, nw_last_row):
                if threshed[y, x] == 255:
                    return x
    else:
        for x in range(nw_index[0] + 1, col_range[1]):
            is_space_col = 1
            for y in range(nw_first_row, nw_last_row):
                if threshed[y, x] == 255:
                    is_space_col = 0
                    break
            if is_space_col == 1:
                space_sum += 1
                if space_sum > (nw_space_avg*avg_multiplier):
                    return x - space_sum + 1
            else:
                space_sum = 0

    return -1


#return the average space between contours in current line. space between words >= avg_space
def get_avg_space():
    space_sum = 0
    contours_number = 0
    prev_col = -1
    curr_col = 0

    for cont in contours:
        if nw_first_row <= cont[0][0][1] <= nw_last_row:
            contours_number += 1
            for x in cont:
                if x[0][0] > curr_col:
                    curr_col = x[0][0]
            if prev_col != -1:
                space_sum += (curr_col - prev_col)
            prev_col = curr_col
            curr_col = 0
        elif cont[0][0][1] > last_text_row:
            break

    avg = round(space_sum / contours_number)
    #print("avg", avg)

    return avg



def get_word_col():
    global nw_index, nw_first_row, nw_last_row
    col_range = find_col_text_range(nw_first_row, nw_last_row)

    #no more text line
    if col_range[0] == -1:
        return -1

    #first execution
    if nw_index[0] == -1:
        nw_index[0] = col_range[0]
    else:
        nw_index[0] = get_next_col(1, col_range)
        #no more words in row
        if nw_index[0] == -1:
            nw_first_row = find_next_line_row(flat, nw_last_row)
            nw_last_row = find_next_blank_row(threshed, nw_first_row)
            nw_index = [-1, -1]
            get_word_col()

    nw_index[1] = get_next_col(0, col_range)
    if nw_index[1] == -1:
        nw_index[1] = col_range[1]

    return 0


def draw_word_contour(image):
    word_contour = get_partial_line_contour(nw_first_row, nw_last_row, nw_index)
    add_convexHull(image, word_contour)

    return


#the result of this function doesn't save automatically. Draw word after word from current line to the end of text
def find_next_word():
    global res, nw_first_row, nw_last_row, nw_index, nw_space_avg
    output_label["text"] = ""

    if err == 1:
        set_error(err, "Please enter image before")
    elif first_text_row == -1 or nw_first_row == -1:
        set_error(0, "No more words in image")
    else:
        #path = 'C:/Users/lior_/PycharmProjects/final text line extraction/temp'
        path = os.path.join(os.path.dirname(__file__), 'temp')


        #not found line yet
        if last_text_row == -1 and nw_index == -1:
            nw_first_row = find_next_line_row(flat, 0)
            nw_last_row = find_next_blank_row(threshed, nw_first_row)
            nw_index = [-1, -1]
            nw_space_avg = get_avg_space()
            cv2.imwrite(os.path.join(path, "before_nw.png"), res)
        #not a sequence of next_line
        elif nw_first_row == -2:
            nw_first_row = first_text_row
            nw_last_row = last_text_row
            nw_index = [-1, -1]
            nw_space_avg = get_avg_space()
            cv2.imwrite(os.path.join(path, "before_nw.png"), res)
            res = cv2.imread(os.path.join(path, "temp_image.png"), cv2.IMREAD_GRAYSCALE)

        #find word's col range
        if get_word_col() == -1:
            set_error(0, "No more words in image")
        else:
            draw_word_contour(res)

        show_image("Next word result", res)

    return



class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget
        self.master.title("GUI menu")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a button instance
        quitButton = Button(self, text="Quit", command=quit_program, bg="light blue", height="2", width="14", activebackground="light yellow")
        # placing the button on my window
        quitButton.place(x=555, y=5)

        next_lineButton = Button(self, text="Find next line", command=find_line, bg="light blue", height="2", width="14", activebackground="light yellow")
        next_lineButton.place(x=115, y=5)

        improve_accuracyButton = Button(self, text="Improve accuracy", command=improve_accuracy, bg="light blue", height="2", width="14", activebackground="light yellow")
        improve_accuracyButton.place(x=225, y=5)

        enter_imageButton = Button(self, text="Enter image",command=enter_image, bg="light blue", height="2", width="14", activebackground="light yellow")
        enter_imageButton.place(x=5, y=5)

        saveButton = Button(self, text="Save result!", command=save_image, bg="light blue", height="2",width="14", activebackground="light yellow")
        saveButton.place(x=335, y=5)

        next_wordButton = Button(self, text="Next word", command=find_next_word, bg="light blue", height="2", width="14", activebackground="light yellow")
        next_wordButton.place(x=445, y=5)

        global input_entry, output_label
        output_label = Label(self, font=(1))
        output_label.place(x=5, y=60)

        Label(self, text="Text input").place(x=5, y=95)
        input_entry = Entry(self, width=25)
        input_entry.place(x=70, y=95)
        input_entry.focus()



#create directories if does not exists
def create_directory(directory_path):

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return

directory1 = os.path.join(os.path.dirname(__file__), 'temp')
directory2 = os.path.join(os.path.dirname(__file__), 'results')
create_directory(directory1)
create_directory(directory2)


root = Tk()

#size of the window
root.geometry("690x130")

app = Window(root)
root.mainloop()


