import numpy as np
import pymol
import chempy
import sys


from pymol.cgo import *
from pymol import cmd
from pymol.vfont import plain
from random import randint

#############################################################################
#
# drawBoundingBox.py -- Draws a box surrounding a selection
#
#
# AUTHOR: Jason Vertrees
# DATE  : 2/20/2009
# NOTES : See comments below.
#
#############################################################################
def drawBoundingBox(extent=None, selection="(all)", padding=0.0, linewidth=2.0, r=1.0, g=1.0, b=1.0):
        """
        DESCRIPTION
                Given selection, draw the bounding box around it.

        USAGE:
                drawBoundingBox [selection, [padding, [linewidth, [r, [g, b]]]]]

        PARAMETERS:
                selection,              the selection to enboxen.  :-)
                                        defaults to (all)

                padding,                defaults to 0

                linewidth,              width of box lines
                                        defaults to 2.0

                r,                      red color component, valid range is [0.0, 1.0]
                                        defaults to 1.0

                g,                      green color component, valid range is [0.0, 1.0]
                                        defaults to 1.0

                b,                      blue color component, valid range is [0.0, 1.0]
                                        defaults to 1.0

        RETURNS
                string, the name of the CGO box

        NOTES
                * This function creates a randomly named CGO box that minimally spans the protein. The
                user can specify the width of the lines, the padding and also the color.
        """
        if not extent:
            ([minX, minY, minZ],[maxX, maxY, maxZ]) = cmd.get_extent(selection)
        else:
            ([minX, minY, minZ],[maxX, maxY, maxZ]) = extent

        print "Box dimensions (%.2f, %.2f, %.2f)" % (maxX-minX, maxY-minY, maxZ-minZ)

        minX = minX - float(padding)
        minY = minY - float(padding)
        minZ = minZ - float(padding)
        maxX = maxX + float(padding)
        maxY = maxY + float(padding)
        maxZ = maxZ + float(padding)

        if padding != 0:
                 print "Box dimensions + padding (%.2f, %.2f, %.2f)" % (maxX-minX, maxY-minY, maxZ-minZ)

        boundingBox = [
                LINEWIDTH, float(linewidth),

                BEGIN, LINES,
                COLOR, float(r), float(g), float(b),

                VERTEX, minX, minY, minZ,       #1
                VERTEX, minX, minY, maxZ,       #2

                VERTEX, minX, maxY, minZ,       #3
                VERTEX, minX, maxY, maxZ,       #4

                VERTEX, maxX, minY, minZ,       #5
                VERTEX, maxX, minY, maxZ,       #6

                VERTEX, maxX, maxY, minZ,       #7
                VERTEX, maxX, maxY, maxZ,       #8


                VERTEX, minX, minY, minZ,       #1
                VERTEX, maxX, minY, minZ,       #5

                VERTEX, minX, maxY, minZ,       #3
                VERTEX, maxX, maxY, minZ,       #7

                VERTEX, minX, maxY, maxZ,       #4
                VERTEX, maxX, maxY, maxZ,       #8

                VERTEX, minX, minY, maxZ,       #2
                VERTEX, maxX, minY, maxZ,       #6


                VERTEX, minX, minY, minZ,       #1
                VERTEX, minX, maxY, minZ,       #3

                VERTEX, maxX, minY, minZ,       #5
                VERTEX, maxX, maxY, minZ,       #7

                VERTEX, minX, minY, maxZ,       #2
                VERTEX, minX, maxY, maxZ,       #4

                VERTEX, maxX, minY, maxZ,       #6
                VERTEX, maxX, maxY, maxZ,       #8

                END
        ]

        boxName = "box_" + str(randint(0,10000))
        while boxName in cmd.get_names():
                boxName = "box_" + str(randint(0,10000))

        cmd.load_cgo(boundingBox,boxName)
        return boxName

cmd.extend ("drawBoundingBox", drawBoundingBox)

class PutCenterCallback(object):
    prev_v = None

    def __init__(self, name, corner=0):
        self.name = name
        self.corner = corner
        self.cb_name = cmd.get_unused_name('_cb')

    def load(self):
        cmd.load_callback(self, self.cb_name)

    def __call__(self):
        if self.name not in cmd.get_names('objects'):
            import threading
            threading.Thread(None, cmd.delete, args=(self.cb_name,)).start()
            return

        v = cmd.get_view()
        if v == self.prev_v:
            return
        self.prev_v = v

        t = v[12:15]

        if self.corner:
            vp = cmd.get_viewport()
            R_mc = [v[0:3], v[3:6], v[6:9]]
            off_c = [0.15 * v[11] * vp[0] / vp[1], 0.15 * v[11], 0.0]
            if self.corner in [2,3]:
                off_c[0] *= -1
            if self.corner in [3,4]:
                off_c[1] *= -1
            off_m = cpv.transform(R_mc, off_c)
            t = cpv.add(t, off_m)

        z = -v[11] / 30.0
        m = [z, 0, 0, t[0] / z, 0, z, 0, t[1] / z, 0, 0, z, t[2] / z, 0, 0, 0, 1]
        cmd.set_object_ttt(self.name, m, homogenous=1)

def axes(name='axes'):
    '''
DESCRIPTION

    Puts coordinate axes to the lower left corner of the viewport.
    '''
    from pymol import cgo

    cmd.set('auto_zoom', 0)

    w = 0.06 # cylinder width
    l = 0.75 # cylinder length
    h = 0.25 # cone hight
    d = w * 1.618 # cone base diameter

    obj = [cgo.CYLINDER, 0.0, 0.0, 0.0,   l, 0.0, 0.0, w, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
           cgo.CYLINDER, 0.0, 0.0, 0.0, 0.0,   l, 0.0, w, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
           cgo.CYLINDER, 0.0, 0.0, 0.0, 0.0, 0.0,   l, w, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
           cgo.CONE,   l, 0.0, 0.0, h+l, 0.0, 0.0, d, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
           cgo.CONE, 0.0,   l, 0.0, 0.0, h+l, 0.0, d, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
           cgo.CONE, 0.0, 0.0,   l, 0.0, 0.0, h+l, d, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    PutCenterCallback(name, 1).load()
    cmd.load_cgo(obj, name)

def show_boxfile(f, boxshape, resolution, proplist=None, dtype=np.float32):
    if proplist and len(proplist) != boxshape[3]:
        raise ValueError("Property list does not match specified boxshape")

    with open(f, "rb") as infile:
        file_array = np.frombuffer(infile.read(), dtype=dtype)

    box = file_array.reshape([
        boxshape[0],
        boxshape[1],
        boxshape[2],
        boxshape[3]])

    chainid = ""
    vdw = 1.4

    shape_array = np.array(boxshape[0:3])

    # create meshgrid with coordinates

    for property_index in range(box.shape[3]):
        model = chempy.models.Indexed()
        if proplist:
            modelname = proplist[property_index]
        else:
            modelname = "prop%d" % property_index

        for grid_coord_0 in xrange(boxshape[0]):
            for grid_coord_1 in xrange(boxshape[1]):
                for grid_coord_2 in xrange(boxshape[2]):
                    grid_coord = np.array([grid_coord_0, grid_coord_1, grid_coord_2])
                    coord = (grid_coord - (shape_array / 2.0) + 0.5) * resolution

                    val = box[grid_coord_0, grid_coord_1, grid_coord_2, property_index]

                    if np.equal(val, 0.0):
                        pass
                        # continue

                    atom = chempy.Atom()
                    atom.coord = coord
                    atom.b = val
                    atom.chain = chainid
                    atom.hetatm = 0
                    atom.vdw = vdw
                    model.add_atom(atom)

        pymol.cmd.load_model(model, modelname)
        pymol.cmd.hide("everything", modelname)
        pymol.cmd.set("sphere_scale", 0.1, modelname)
        pymol.cmd.spectrum("b", selection=modelname)
        pymol.cmd.set("transparency", 0.5, modelname)

        pymol.cmd.show("spheres", modelname)

        pymol.cmd.select("zeros", "! (b>0 | b<0)")

    if proplist:
        pymol.cmd.group("Box", proplist)
    else:
        pymol.cmd.group("Box", "prop*")
    
    shapehalf = (shape_array / 2.0) * resolution

    drawBoundingBox(extent=([-shapehalf[0], -shapehalf[1], -shapehalf[2]], [shapehalf[0], shapehalf[1], shapehalf[2]]), r=0.0, g=0.0, b=0.0)
    axes()

    pymol.cmd.set('bg_rgb',0,'',0)

BOX_SUFFIX = '.box'
if __name__ == "pymol":
    args = sys.argv[1:]

    if len(args) < 3:
         print >> sys.stderr, "Error: at least three arguments required"
    else:
        show_boxfile(args[-3], [int(s) for s in args[-2].split(",")], float(args[-1]))

        if args[-3].endswith(BOX_SUFFIX):
            boxfile_name = args[-3][:-len(BOX_SUFFIX)]
        else:
            boxfile_name = args[-3]


        pymol.cmd.save(boxfile_name + ".pse")

