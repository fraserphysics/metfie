'''cmf_models.py defines classes for representing model choices that keep
track of provenance.
'''
class Provenance:
    '''A tree structure designed to track the origins of a Component
    instance.
    '''
    def __init__(
            self,
            frame,        # inspect.stack() reference to code that called
            text='',      # Explanatory text
            branches=None,# A list or tuple of source Components
            max_hist=5
            ):
        self.file = frame[1]             # Name of file
        self.line_num = frame[2]         # Line number of file
        self.line = frame[4][0].strip()  # text of the specified line
        self.text = text
        if branches == None:
            self.depth = 0
            self.branches = ()
        else:
            self.depth = max(x.depth for x in branches) + 1
            self.branches = tuple(branches)
            assert self.depth <= max_hist,'''max hist exceeded
Printing self:
{0}'''.format(str(self))
    def _indent(self, root_depth):
        '''return a string description of self indented by (root_depth -
        self.depth).  The method calls itself to indent each branch of
        the provenance tree by depth.
        '''
        space = (root_depth - self.depth)*4*' '
        lines = self.text.splitlines()
        rv = '\n'+space+'depth={0:d}, {1} line {2:d}:{3}\n'.format(
            self.depth, self.file, self.line_num, self.line)
        for line in lines[1:]:
            rv += space+line+'\n'
        for branch in self.branches:
            rv += branch._indent(root_depth)
        return rv
    def __str__(self):
        return self._indent(self.depth)
    def html(
            self,       # Provenance instance
            name,       # Component name qualified by branch indices
            ):
        '''Add to html view of self.
        '''
        from markup import oneliner
        import os.path

        # Start with a description list key
        return_value = oneliner.dt(name,id=name)+'\n'

        # Construct text for inclusion as description list value
        href = 'file://'+self.file # EG, "file:///absolute_path/calc.py"
        text = '{0}\n line {1:d}: {2}\n'.format(
                oneliner.a(os.path.basename(self.file),href=href),
                self.line_num,
                self.line)
        if self.text != '':
            text += oneliner.p(self.text)+'\n'
        if len(self.branches) == 0:
            text += oneliner.p('No more provenance of {0}'.format(name))+'\n'
        else:
            text += 'Parent Components: '
            for i,branch in enumerate(self.branches):
                name_ = '{0}.{1}'.format(name,i)
                text += oneliner.a(name_,href='#{0}'.format(name_))+', '

        # Insert description list value
        return_value += oneliner.dd(text.rstrip(' ,'))

        # Recursion on branches
        for i,branch in enumerate(self.branches):
            return_value += branch.html('{0}.{1}'.format(name,i))
        return return_value

class Component:
    '''Parent class intended for subclasses.  A Component object represents a
    decision about an aspect of a model for simulation.  Examples
    include material properties like an entire EOS or just a heat
    capacity.
    
    '''
    def __init__(
        self,           # Component instance
        value,          # Subclasses may restrict this
        comment='',     # String of explanation
        provenance=None,
        max_hist=5      # Prevent huge provenance from loop
        ):

        assert not type(self) == Component,'only use subclasses of Component'
        self.value = value
        self.max_hist = max_hist
        if provenance == None: # Probably a leaf
            from inspect import stack
            self.provenance = Provenance(
                stack()[2], # Context that called for Component sub_class
                comment,
                max_hist=self.max_hist)
        else:
            self.provenance = provenance
        if not hasattr(self, 'display'):
            self.display = False
    def __join__(self, value, comment, branches, depth=2):
        '''return a new instance that is the result of an operation on
         the objects in branches
        '''
        from inspect import stack
        provenance = Provenance(
            stack()[depth], # Context that called for combination
            comment,
            branches,
            max_hist=self.max_hist)
        
        # The following gets a new instance of same class as self even
        # if self is an instance of a subclass
        return self.__class__(value, comment, provenance,
                              max_hist=self.max_hist)
    def __str__(self):
        rv= '''{0}, value = {1}, provenance:
{2}'''.format(self.__class__, self.value, self.provenance)
        return rv
    def html(
            self, # Component instance
            name, # Component name
            ):
        '''Return tuple containing:
             
             1. html description of self suitable for page.li()
             
             2. html description of provenance sutible for page.add
                inside of page.dl(), page.dl.close() pair
        '''
        from markup import oneliner
        if self.display == False:
            return '{0}: value={1}, {2}'.format(
                name,
                self.value,
                oneliner.a('provenance',href='#{0}'.format(name))
                )
        return '{0}: {1}, {2}'.format(
            name,
            oneliner.a('value_link', href='#value of {0}'.format(name)),
            oneliner.a('provenance',href='#{0}'.format(name))
            )
class Float(Component):
    '''
    '''
    def __init__(self, *args, **kwargs):
        import numbers
        assert isinstance(args[0], numbers.Number)
        Component.__init__(self, *args, **kwargs)
    def __arithmetic__(self, x, text, op):
        import numbers
        branches = [self.provenance]
        if isinstance(x, Component):
            x_value = x.value
            branches.append(x.provenance)
            comment = '{0} of two Floats'.format(text)
        else:
            x_value = x
            comment = '{0} of Float and number without provenance'.format(
                text)
        assert isinstance(x_value, numbers.Number)
        return self.__join__(op(self.value,x_value), comment, branches,
            depth=3)
    def __neg__(self):
        branches = [self.provenance]
        comment = 'Negation of Float'
        return self.__join__(-self.value, comment, branches)
    def __truediv__(self, x): # python3
        op = lambda a,b: a/b
        return self.__arithmetic__(x, 'Division', op)
    def __div__(self, x): # python2
        op = lambda a,b: a/b
        return self.__arithmetic__(x, 'Division', op)
    def __mul__(self, x):
        op = lambda a,b: a*b
        return self.__arithmetic__(x, 'Multiplication', op)
    def __add__(self, x):
        op = lambda a,b: a+b
        return self.__arithmetic__(x, 'Addition', op)
    def __sub__(self, x):
        op = lambda a,b: a-b
        return self.__arithmetic__(x, 'Subtraction', op)

def make_html(component_dict, sorted_names=None, title='Simulation'):
    '''Returns a markup.page instance suitable for writing to an html file.
    '''
    import markup
    if sorted_names == None:
        sorted_names = sorted(component_dict.keys())
    page = markup.page()
    page.h1('The components of {0}'.format(title))
    page.ul()
    for name in sorted_names:
        page.li(component_dict[name].html(name))
    page.ul.close()
    page.br( )
    
    page.h1('Provenance of components')
    page.dl()
    for name in sorted_names:
        c = component_dict[name]
        page.add(c.provenance.html(name))
    page.dl.close()
    page.br( )
    
    page.h1('Extended displays of component values')
    page.dl()
    for name in sorted_names:
        c = component_dict[name]
        if c.display == False:
            continue
        key = 'value of {0}'.format(name)
        page.add(markup.oneliner.dt(key,id=key)+'\n'+
                 markup.oneliner.dd(c.display())
                 )
    page.dl.close()
    return page
def demo():
    ''' Fit an eos and write an html file
    '''
    import calc

    vt = calc.experiment()
    fit,e = calc.best_fit(vt)
    make_page(fit, 'gun.html')
    return 0
def make_page(gun,file_name):
    keys = sorted(gun.components)
    component_dict = dict((key,getattr(gun, key)) for key in keys)
    page = make_html(component_dict, keys, 'Simulated Gun')
    file_ = open(file_name,'wt')
    file_.write(page.__str__())
    return
def test():
    import calc
    gun = calc.GUN()
    make_page(gun,'test.html')
    return 0
    print("provenance of the components of simulated gun:")
    for key in sorted(gun.components):
        print('component {0}: {1}'.format(key, getattr(gun, key)))
    return 0
    comment = '''This is a long
multi line explanation
of the value 2.7
which is b in test'''
    a = Float(1)
    b = Float(27,comment)
    c = a+b
    d = c + 3
    print('''
a=%s
-----------------------------------------------------------------
a+b=%s
-----------------------------------------------------------------
c+3=%s
    '''%(str(a), str(c), str(d)))

    return 0
if __name__ == "__main__":
    demo()
    #test()

#---------------
# Local Variables:
# mode: python
# End:
