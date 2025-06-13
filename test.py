from omni.isaac.core import World
world = World()
for prim in world.scene.stage.Traverse():
    print(prim.GetPath())
