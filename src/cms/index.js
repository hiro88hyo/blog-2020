// @flow strict
import CMS from 'decap-cms-app';
import PagePreview from './preview-templates/page-preview';
import PostPreview from './preview-templates/post-preview';

CMS.registerPreviewTemplate('pages', PagePreview);
CMS.registerPreviewTemplate('posts', PostPreview);
