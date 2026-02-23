'use strict';

const React = require('react');
const siteConfig = require('../config.js');

const onRenderBody = ({ setHeadComponents }) => {
  const { useKatex } = siteConfig;

  if (useKatex) {
    setHeadComponents([
      React.createElement('link', {
        key: 'katex-stylesheet',
        rel: 'stylesheet',
        href: '/css/katex/katex.min.css'
      })
    ]);
  }
};

module.exports = onRenderBody;
