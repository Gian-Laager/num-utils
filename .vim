nnoremap <leader><leader>c :T cmake -B cmake-build-debug -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1  && rm -f compile_commands.json && ln -s cmake-build-debug/compile_commands.json ./compile_commands.json<CR>
nnoremap <leader><leader>C :T cmake -B cmake-build-release -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1  && rm -f compile_commands.json && ln -s cmake-build-release/compile_commands.json ./compile_commands.json<CR>
nnoremap <leader><leader>b :T cmake --build cmake-build-debug -j 17 --target num-utils-test<CR>
nnoremap <leader><leader>B :T cmake --build cmake-build-release --j 17 -target num-utils-test<CR>
" nnoremap <leader><leader>r :T cmake --build cmake-build-debug --target num-utils && ./cmake-build-debug/num-utils<CR>
" nnoremap <leader><leader>R :T cmake --build cmake-build-release --target num-utils && ./cmake-build-release/num-utils<CR>

nnoremap <leader><leader>t :T cmake --build cmake-build-debug -j 17 --target num-utils-test && ./cmake-build-debug/test/num-utils-test<CR>
nnoremap <leader><leader>T :T cmake --build cmake-build-release -j 17 --target num-utils-test && ./cmake-build-release/test/num-utils-test<CR>


lua<<EOF
local dap = require('dap')
dap.adapters.lldb = {
  type = 'executable',
  command = '/usr/bin/lldb-dap', -- adjust as needed, must be absolute path
  name = 'lldb'
}

dap.configurations.cpp = {
  {
    name = 'Launch',
    type = 'lldb',
    request = 'launch',
    program = 'cmake-build-debug/test/num-utils-test',
    cwd = '${workspaceFolder}',
    stopOnEntry = false,
    args = {},

    -- 💀
    -- if you change `runInTerminal` to true, you might need to change the yama/ptrace_scope setting:
    --
    --    echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
    --
    -- Otherwise you might get the following error:
    --
    --    Error on launch: Failed to attach to the target process
    --
    -- But you should be aware of the implications:
    -- https://www.kernel.org/doc/html/latest/admin-guide/LSM/Yama.html
    runInTerminal = true,
  },
}
