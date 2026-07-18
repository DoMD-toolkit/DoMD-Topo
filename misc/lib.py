class Config(object):
    def __init__(self, reactant_config,
                 reaction_template,
                 rigid_config,
                 box_tensor,
                 cg_sys,
                 reaction_list,
                 cg_graphs=None):
        self.reactant_config = reactant_config
        self.reaction_template = reaction_template
        self.rigid_config = rigid_config
        self.cg_sys = cg_sys
        self.cg_graphs = cg_graphs
        self.reaction_list = reaction_list
        self.box_tensor = box_tensor

    def __str__(self):
        return (f"Config(reactant_config={self.reactant_config}, "
                f"reaction_template={self.reaction_template}, "
                f"rigid_config={self.rigid_config}, "
                f"box_tensor={self.box_tensor}, "
                f"cg_sys={self.cg_sys}, "
                f"reaction_list={self.reaction_list}, "
                f"cg_graphs={self.cg_graphs})")
