/**
 * 登录/注册页面 v3
 * 包含表单验证和 API 调用
 */

// ========== 全局状态 ==========
let passwordVisible = false;
let isCheckingAuth = false; // 防止重复检查
let lastRedirectTime = 0;
const MIN_REDIRECT_INTERVAL = 3000; // 3秒内不重复跳转

// ========== DOM 元素 ==========
const elements = {
    // 表单
    loginForm: document.getElementById('loginForm'),
    loginContent: document.getElementById('loginContent'),

    // 输入框
    loginUsername: document.getElementById('loginUsername'),
    loginPassword: document.getElementById('loginPassword'),

    // 按钮
    loginBtn: document.getElementById('loginBtn'),
    toggleLoginPassword: document.getElementById('toggleLoginPassword'),
    themeToggle: document.getElementById('themeToggle'),

    // 错误提示
    loginError: document.getElementById('loginError'),

    // 弹窗
    passwordModal: document.getElementById('passwordModal'),
    modalError: document.getElementById('modalError'),
    changePasswordBtn: document.getElementById('changePasswordBtn'),
};

// ========== 表单交互 ==========

function applyTheme(theme) {
    const nextTheme = theme === 'dark' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', nextTheme);
    localStorage.setItem('theme', nextTheme);
    if (elements.themeToggle) {
        const isDark = nextTheme === 'dark';
        elements.themeToggle.setAttribute('aria-label', isDark ? '切换浅色模式' : '切换深色模式');
        elements.themeToggle.setAttribute('title', isDark ? '切换浅色模式' : '切换深色模式');
        const icon = elements.themeToggle.querySelector('.theme-icon');
        if (icon) icon.textContent = isDark ? '☀' : '☾';
    }
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    applyTheme(currentTheme === 'dark' ? 'light' : 'dark');
}

// 密码可见切换
function setupPasswordToggle(btn, inputId) {
    if (!btn) return;

    const input = document.getElementById(inputId);
    const iconEye = btn.querySelector('.icon-eye');
    const iconEyeOff = btn.querySelector('.icon-eye-off');

    btn.addEventListener('click', () => {
        const isVisible = input.type === 'text';
        input.type = isVisible ? 'password' : 'text';

        iconEye?.classList.toggle('hidden', !isVisible);
        iconEyeOff?.classList.toggle('hidden', isVisible);
    });
}

// 输入框聚焦效果
function setupInputFocus() {
    const inputs = document.querySelectorAll('input[type="text"], input[type="email"], input[type="password"]');
    inputs.forEach((input) => {
        input.addEventListener('focus', () => {
            // 添加聚焦效果
            input.parentElement.classList.add('input-focused');
        });

        input.addEventListener('blur', () => {
            // 移除聚焦效果
            input.parentElement.classList.remove('input-focused');
        });
    });
}

// 添加输入框聚焦样式
const style = document.createElement('style');
style.textContent = `
    .password-wrapper.input-focused, .form-group.input-focused {
        transform: translateY(-2px);
        transition: transform 0.3s ease;
    }
`;
document.head.appendChild(style);

// ========== API 调用 ==========

// 显示错误信息
function showError(element, message) {
    if (!element) return;
    element.textContent = message;
    element.classList.add('show');
    setTimeout(() => element.classList.remove('show'), 5000);
}

// 设置按钮加载状态
function setLoading(btn, loading) {
    if (!btn) return;

    const text = btn.querySelector('.btn-text');
    const arrow = btn.querySelector('.btn-arrow');
    const loader = btn.querySelector('.btn-loader');

    btn.disabled = loading;

    if (loading) {
        text?.classList.add('hidden');
        arrow?.classList.add('hidden');
        loader?.classList.remove('hidden');
    } else {
        text?.classList.remove('hidden');
        arrow?.classList.remove('hidden');
        loader?.classList.add('hidden');
    }
}

// 登录
async function handleLogin(e) {
    e.preventDefault();

    const username = elements.loginUsername?.value.trim();
    const password = elements.loginPassword?.value;

    if (!username || !password) {
        showError(elements.loginError, '请填写用户名和密码');
        return;
    }

    setLoading(elements.loginBtn, true);

    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password }),
        });

        const data = await response.json();

        if (data.success) {
            localStorage.setItem('token', data.token);
            localStorage.setItem('user', JSON.stringify(data.user));

            // 检查是否需要修改默认密码
            if (!data.user.password_changed && data.user.role === 'admin') {
                elements.passwordModal?.classList.remove('hidden');
            } else {
                lastRedirectTime = Date.now();
                window.location.href = '/chat';
            }
        } else {
            showError(elements.loginError, data.message || '登录失败');
        }
    } catch (error) {
        showError(elements.loginError, '网络错误，请检查连接');
    } finally {
        setLoading(elements.loginBtn, false);
    }
}

// 修改密码
async function handleChangePassword() {
    const oldPassword = document.getElementById('oldPassword')?.value;
    const newPassword = document.getElementById('newPassword')?.value;
    const confirmPassword = document.getElementById('confirmNewPassword')?.value;

    if (!oldPassword || !newPassword) {
        showError(elements.modalError, '请填写所有字段');
        return;
    }

    if (newPassword.length < 6) {
        showError(elements.modalError, '新密码至少6位');
        return;
    }

    if (newPassword !== confirmPassword) {
        showError(elements.modalError, '两次输入的密码不一致');
        return;
    }

    try {
        const response = await fetch('/api/auth/change-password', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + localStorage.getItem('token'),
            },
            body: JSON.stringify({
                old_password: oldPassword,
                new_password: newPassword,
            }),
        });

        const data = await response.json();

        if (data.success) {
            alert('密码修改成功！');
            elements.passwordModal?.classList.add('hidden');
            lastRedirectTime = Date.now();
            window.location.href = '/chat';
        } else {
            showError(elements.modalError, data.message || '修改失败');
        }
    } catch (error) {
        showError(elements.modalError, '网络错误');
    }
}

// ========== 初始化 ==========

function init() {
    applyTheme(localStorage.getItem('theme') || 'light');
    elements.themeToggle?.addEventListener('click', toggleTheme);

    // 【暂时禁用自动登录检查，避免循环刷新问题】
    // 如需启用，请取消下面的注释并确保 localStorage 中没有无效 token
    /*
    const token = localStorage.getItem('token');
    if (token && !window.location.pathname.includes('/chat')) {
        fetch('/api/auth/profile', {
            headers: { 'Authorization': 'Bearer ' + token },
        }).then((res) => {
            if (res.ok) {
                window.location.href = '/chat';
            } else {
                localStorage.removeItem('token');
                localStorage.removeItem('user');
            }
        }).catch(() => {
            // 请求失败，不做处理
        });
    }
    */

    // 密码切换
    setupPasswordToggle(elements.toggleLoginPassword, 'loginPassword');

    // 输入框聚焦
    setupInputFocus();

    // 表单提交
    elements.loginForm?.addEventListener('submit', handleLogin);

    // 修改密码
    elements.changePasswordBtn?.addEventListener('click', handleChangePassword);
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', init);
