from audioop import bias
from typing import Dict, List, Optional

from llvmlite import binding, ir
from llvmlite.ir.instructions import CallInstr, ICMPInstr, LoadInstr, PhiInstr

from .astnodes import (
    AssignStmt,
    BinaryExpr,
    BooleanLiteral,
    CallExpr,
    ClassType,
    ExprStmt,
    FuncDef,
    Identifier,
    IfExpr,
    IfStmt,
    IntegerLiteral,
    Node,
    NoneLiteral,
    Program,
    ReturnStmt,
    StringLiteral,
    TypedVar,
    UnaryExpr,
    VarDef,
    WhileStmt,
)
from .backend import Backend


class LLVMBackendError(Exception):
    def __init__(self, message, node: Node = None):
        if node is not None:
            if hasattr(node, "lineno"):
                super().__init__(
                    message
                    + ". Line {:d} Col {:d}".format(node.lineno, node.col_offset)
                )
                return
        super().__init__(message + ".")


class LLVMBackend(Backend):
    def __init__(self, int_bits=32, bool_bits=1, char_bits=8):
        # Create module to hold IR code
        self.module = ir.Module()
        self.module.triple = binding.get_process_triple()

        # Set parameters
        self.int_bits = int_bits
        self.bool_bits = bool_bits
        self.char_bits = char_bits

        # Create main func to hold toplevel declarations and statements
        self.main_func = ir.Function(
            self.module, ir.FunctionType(ir.IntType(self.int_bits), []), "main"
        )
        self.builder: Optional[ir.IRBuilder] = ir.IRBuilder(
            self.main_func.append_basic_block("entry")
        )

        # Declare global functions
        voidptr_ty = ir.PointerType(ir.IntType(self.char_bits))
        printf_ty = ir.FunctionType(
            ir.IntType(self.int_bits), [voidptr_ty], var_arg=True
        )
        ir.Function(self.module, printf_ty, "printf")

        # Create symbol table to store variables in scope
        self.func_symtab: List[Dict[str, ir.Value]] = [{}]

    def visit(self, node: Node):
        return node.visit(self)

    # Performs memory allocation for a new variable
    def _create_alloca(self, name, typ):
        if self.builder is None:
            raise Exception("No builder is active")

        with self.builder.goto_entry_block():
            alloca = self.builder.alloca(typ, size=None, name=name)
        return alloca

    # Returns the address of a variable from the symbol table
    def _get_var_addr(self, name):
        try:
            return self.func_symtab[-1][name]
        except KeyError:
            raise Exception("Undefined variable: " + name)

    # Returns the LLVM type object corresponding to the type name
    def _get_llvm_type(self, typename: str):
        if typename == "int":
            return ir.IntType(self.int_bits)
        elif typename == "str":
            return ir.PointerType(ir.IntType(self.char_bits))
        elif typename == "bool":
            return ir.IntType(self.bool_bits)
        elif typename == "<None>":
            return ir.VoidType()
        else:
            raise Exception(f"Invalid type: {typename}")

    ##################################
    #        TO BE IMPLEMENTED       #
    ##################################

    def VarDef(self, node: VarDef):
        '''
        c_str_val = ir.Constant(ir.ArrayType(ir.IntType(8), len(arg)), bytearray(arg.encode("utf8"))) #creates the c_str_value as a constant
        c_str = builder.alloca(c_str_val.type) #creation of the allocation of the %".2" variable
        builder.store(c_str_val, c_str) #store as defined on the next line below %".2"
        '''
        # print(self.visit(node.var))
        # print(self.visit(node.var)['type'] == self._get_llvm_type('int'))
        '''ty = self.visit(node.var)['type']
        if ty == self._get_llvm_type('int'):
            # self.builder.alloca(self._get_llvm_type('int'), None, self.visit(node.var)['name'])
            # self._create_alloca(self.visit(node.var)['name'], self._get_llvm_type('int'))
            # self.builder.store(self.visit(node.value), ir.IntType.as_pointer(self.visit(node.var)['type']), None)
            # int_val = ir.Constant(self._get_llvm_type('int'), self.visit(node.value))
            """ int_var = self.builder.alloca(self._get_llvm_type('int'), None, self.visit(node.var)['name'])
            self.builder.store(self.visit(node.value), int_var) """
            alloca = self._create_alloca(self.visit(node.var)['name'], self._get_llvm_type('int'))
            self.builder.store(self.visit(node.value), alloca)
            self.func_symtab[-1][self.visit(node.var)['name']] = alloca
        elif ty == self._get_llvm_type('bool'):
            # self.builder.alloca(self._get_llvm_type('bool'), None, self.visit(node.var)['name'])
            # self._create_alloca(self.visit(node.var)['name'], self._get_llvm_type('bool'))
            # self.builder.store(self.visit(node.value), self._get_var_addr(self.visit(node.var)['name']), None)
            # bool_val = ir.Constant(self._get_llvm_type('bool'), self.visit(node.value))
            """ bool_var = self.builder.alloca(self._get_llvm_type('bool'), None, self.visit(node.var)['name'])
            self.builder.store(self.visit(node.value), bool_var) """
            alloca = self._create_alloca(self.visit(node.var)['name'], self._get_llvm_type('bool'))
            self.builder.store(self.visit(node.value), alloca)
            self.func_symtab[-1][self.visit(node.var)['name']] = alloca
        elif ty == self._get_llvm_type('str'):
            # str_const = ir.Constant(self._get_llvm_type('str'), self.visit(node.value))
            # self.builder.alloca(self._get_llvm_type('str'), None, self.visit(node.var)['name'])
            print(self.visit(node.value))
            print(self.visit(node.var))
            print(node.getIdentifier())
            str_val = ir.Constant(ir.ArrayType(ir.IntType(8), len(self.visit(node.value))), bytearray(self.visit(node.value).encode('utf8')))
            str_var = self.builder.alloca(str_val.type)
            self.builder.store(str_val, str_var)
        else:
            # self.builder.alloca(self._get_llvm_type('<None>'), None, self.visit(node.var)['name'])
            void_var = self.builder.alloca(self._get_llvm_type('void'), None, self.visit(node.var)['name'])
            self.builder.store(self.visit(node.value), void_var)'''
        alloca = self._create_alloca(self.visit(node.var)['name'], self.visit(node.var)['type'])
        self.builder.store(self.visit(node.value), alloca)
        self.func_symtab[-1][self.visit(node.var)['name']] = alloca

    def AssignStmt(self, node: AssignStmt):
        """ for t in node.targets:
            self.builder.store(node.value, ir.IntType.as_pointer()) """
            # self.builder.store(self.visit(node.value), alloca)
        # print("in assgn")
        # print(self.func_symtab)
        # print(self.func_symtab[0][node.targets[0].name])
        # print(self.visit(node.value))
        # print(type(node.value))
        # print("in assstmt")
        # if type(node.value) == BinaryExpr:
        #     print(node.value.operator)
        print(self.visit(node.value))
        print(self.func_symtab)
        for t in node.targets:
            print(t.name)
            alloca = self.func_symtab[-1][t.name]
            self.builder.store(self.visit(node.value), alloca)
        # print(node.value.left.name)
        # if node.value.operator == '+':
            # self.builder.store(node.value, self.func_symtab[0][node.targets[0].name], None)
            # print(self.visit(node.value.left))

    def IfStmt(self, node: IfStmt):
        if_condition = self.builder.append_basic_block(self.module.get_unique_name("if.condition"))
        if_body = self.builder.append_basic_block(self.module.get_unique_name("if.body"))
        else_body = self.builder.append_basic_block(self.module.get_unique_name("else.body"))
        if_else_end = self.builder.append_basic_block(self.module.get_unique_name("if_else.end"))

        self.builder.branch(if_condition)

        with self.builder.goto_block(if_condition):
            condition = self.visit(node.condition)
            self.builder.cbranch(condition, if_body, else_body)

        with self.builder.goto_block(if_body):
            for stmt in node.thenBody:
                self.visit(stmt)
            self.builder.branch(if_else_end)
        
        with self.builder.goto_block(else_body):
            for stmt in node.elseBody:
                self.visit(stmt)

        self.builder.position_at_end(if_else_end)

    def WhileStmt(self, node: WhileStmt):
        if self.builder is None:
            raise Exception("No builder is active")

        bb_condition = self.builder.append_basic_block(self.module.get_unique_name("while.condition"))
        bb_body = self.builder.append_basic_block(self.module.get_unique_name("while.body"))
        bb_end = self.builder.append_basic_block(self.module.get_unique_name("while.end"))

        self.builder.branch(bb_condition)

        with self.builder.goto_block(bb_condition):
            condition = self.visit(node.condition)
            self.builder.cbranch(condition, bb_body, bb_end)

        with self.builder.goto_block(bb_body):
            for stmt in node.body:
                self.visit(stmt)
            self.builder.branch(bb_condition)

        self.builder.position_at_end(bb_end)

    def BinaryExpr(self, node: BinaryExpr) -> Optional[ICMPInstr]:
        # print(node.left)
        # print("in binexpr")
        left = self.visit(node.left)
        right = self.visit(node.right)
        print("left", left)
        print("right", right)
        if node.operator in ['+', '-', '*', '%']:
            # print('arithmetic')
            if node.operator == '+':
                self.builder.add(left, right, self.module.get_unique_name('add_temp'))
            elif node.operator == '-':
                self.builder.sub(left, right, self.module.get_unique_name('sub_temp'))
            elif node.operator == '*':
                self.builder.mul(left, right, self.module.get_unique_name('mul_temp'))
            else:
                self.builder.srem(left, right, self.module.get_unique_name('mod_temp'))
        elif node.operator in ['and', 'or']:
            # print('logical')
            if node.operator == 'and':
                self.builder.and_(left, right, self.module.get_unique_name('and_temp'))
            else:
                self.builder.or_(left, right, self.module.get_unique_name('or_temp'))
        elif node.operator in ['>', '<', '<=', '>=', '==', '!=']:
            # print('relational')
            return self.builder.icmp_signed(node.operator, left, right, self.module.get_unique_name('icmp_temp'))

    def Identifier(self, node: Identifier) -> LoadInstr:
        # print(self.func_symtab[0][node.name])
        return self.builder.load(self._get_var_addr(node.name))
        # print(node.visit(node))
        # print('in id')

    def IfExpr(self, node: IfExpr) -> PhiInstr:
        pass

    ##################################
    #      END OF IMPLEMENTATION     #
    ##################################

    # TOP LEVEL & DECLARATIONS
    def Program(self, node: Program):
        for d in node.declarations:
            self.visit(d)
        for s in node.statements:
            self.visit(s)

        # Find the exit basic block and terminate it
        for bb in self.main_func.basic_blocks:
            if not bb.is_terminated:
                self.builder = ir.IRBuilder(bb)
                self.builder.position_at_end(bb)
                self.builder.ret(self._get_llvm_type("int")(0))
        return self

    def FuncDef(self, node: FuncDef):
        # Create new symbol table
        self.func_symtab.append({})

        funcname = node.name.name
        returnType = self._get_llvm_type(node.returnType.className)
        paramTypes = [self.visit(i)["type"] for i in node.params]
        functype = ir.FunctionType(returnType, paramTypes)

        if funcname in self.module.globals:  # Definition for already declared function
            func = existing_func = self.module.globals[funcname]
            if not isinstance(existing_func, ir.Function):
                raise LLVMBackendError(f"Name collision: {funcname}", node)
            if not existing_func.is_declaration:
                raise LLVMBackendError(f"Redefinition of {funcname}", node)
            if len(existing_func.function_type.args) != len(functype.args):
                raise LLVMBackendError(
                    f"Declaration and definition of {funcname} have different signatures",
                    node,
                )
        else:  # New function
            func = ir.Function(self.module, functype, funcname)
            for (name, arg) in zip(
                [self.visit(i)["name"] for i in node.params], func.args
            ):
                arg.name = name

        bb_entry = func.append_basic_block("entry")
        old_builder = self.builder
        self.builder = ir.IRBuilder(bb_entry)

        # Add all arguments to the symbol table and create their allocas
        for arg in func.args:
            alloca = self._create_alloca(arg.name, arg.type)
            self.builder.store(arg, alloca)
            self.func_symtab[-1][arg.name] = alloca

        # Generate code for the body and then return the result
        for d in node.declarations:
            self.visit(d)
        for s in node.statements:
            self.visit(s)
        if not bb_entry.is_terminated:
            self.builder.ret_void()

        # End the function scope
        self.func_symtab.pop()
        self.builder = old_builder

    # STATEMENTS
    def ExprStmt(self, node: ExprStmt):
        self.visit(node.expr)

    def ReturnStmt(self, node: ReturnStmt):
        if self.builder is None:
            raise Exception("No builder is active")

        retval = self.visit(node.value)
        self.builder.ret(retval)

    # Expressions
    def UnaryExpr(self, node: UnaryExpr):
        if self.builder is None:
            raise Exception("No builder is active")

        operand = self.visit(node.operand)
        if node.operator == "-":
            return self.builder.neg(operand, "negtmp")
        elif node.operator == "not":
            return self.builder.sub(self._get_llvm_type("bool")(1), operand)
        else:
            raise LLVMBackendError(f"Unsupported unary operator: {node.operator}", node)

    def CallExpr(self, node: CallExpr) -> CallInstr:
        if self.builder is None:
            raise Exception("No builder is active")

        callee_func = self.module.globals.get(node.function.name, None)
        if callee_func is None or not isinstance(callee_func, ir.Function):
            raise LLVMBackendError(
                f"Call to unknown function {node.function.name}", node
            )

        call_args = [self.visit(arg) for arg in node.args]
        return self.builder.call(callee_func, call_args, "calltmp")

    # LITERALS

    def BooleanLiteral(self, node: BooleanLiteral) -> ir.Constant:
        return self._get_llvm_type("bool")(int(node.value))

    def IntegerLiteral(self, node: IntegerLiteral) -> ir.Constant:
        return self._get_llvm_type("int")(node.value)

    def NoneLiteral(self, node: NoneLiteral) -> ir.Constant:
        return ir.Constant(ir.PointerType(node.value), node.value)

    def StringLiteral(self, node: StringLiteral) -> ir.Constant:
        global_lit = ir.ArrayType(ir.IntType(self.char_bits), len(node.value) + 1)(
            bytearray(node.value.encode("utf8")) + bytearray("\0".encode("utf8"))
        )
        global_name = self.module.get_unique_name("str")
        g = ir.GlobalVariable(self.module, global_lit.type, global_name)
        g.global_constant = True
        g.linkage = "internal"
        g.initializer = global_lit
        return g.gep((ir.IntType(32)(0), ir.IntType(32)(0)))

    # TYPES

    def TypedVar(self, node: TypedVar) -> dict:
        return {
            "name": node.identifier.name,
            "type": self._get_llvm_type(node.type.className),
        }

    def ClassType(self, _: ClassType):
        pass
